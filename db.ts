/**
 * Session Log Indexer — Database Layer
 *
 * SQLite database with FTS5 + vector search support.
 * Handles schema creation, CRUD operations, and search.
 */

import Database from 'better-sqlite3';

/** Type alias for the database instance. */
export type SessionDb = Database.Database;
import path from 'path';
import fs from 'fs';
import * as sqliteVec from '@photostructure/sqlite-vec';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChunkRow {
  id: string;
  session_id: string;
  session_ts: string;
  cwd: string | null;
  model: string | null;
  role: string;
  turn_index: number;
  chunk_index: number;
  token_count: number;
  text: string;
  fts_id: number;
}

export interface ChunkVectorRow {
  chunk_id: string;
  embedding: Buffer;
  model: string;
}

export interface SearchResult {
  sessionId: string;
  sessionTimestamp: string;
  cwd: string | null;
  model: string | null;
  role: string;
  turnIndex: number;
  chunkIndex: number;
  tokenCount: number;
  text: string;
  score: number;
}

export interface IndexedSessionInfo {
  sessionId: string;
  timestamp: string;
  cwd: string | null;
  model: string | null;
  chunks: number;
}

export interface IndexStats {
  totalSessions: number;
  totalChunks: number;
  totalTokens: number;
  indexedSessions: IndexedSessionInfo[];
}

export interface SearchFilters {
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
  role?: string;
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA_SQL = `
-- Core chunks table
CREATE TABLE IF NOT EXISTS session_chunks (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    session_ts  TEXT,
    cwd         TEXT,
    model       TEXT,
    role        TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text        TEXT NOT NULL,
    fts_id      INTEGER UNIQUE
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS session_chunks_fts
    USING fts5(text, content='session_chunks', content_rowid='fts_id');

-- Vector embeddings (768-dim float32 for nomic-embed-text-v1.5)
CREATE TABLE IF NOT EXISTS session_chunk_vectors (
    chunk_id  TEXT PRIMARY KEY REFERENCES session_chunks(id),
    embedding BLOB NOT NULL,
    model     TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_session ON session_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_cwd ON session_chunks(cwd);
CREATE INDEX IF NOT EXISTS idx_chunks_ts ON session_chunks(session_ts);

-- FTS5 sync triggers
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON session_chunks BEGIN
    INSERT INTO session_chunks_fts(rowid, text)
    VALUES (new.fts_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON session_chunks BEGIN
    DELETE FROM session_chunks_fts WHERE rowid = old.fts_id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON session_chunks BEGIN
    UPDATE session_chunks_fts SET text = new.text WHERE rowid = old.fts_id;
END;
`;

// ---------------------------------------------------------------------------
// Database helpers
// ---------------------------------------------------------------------------

/**
 * Encode a float32 array as a Buffer for sqlite-vec.
 */
function encodeEmbedding(vector: number[]): Buffer {
  const buffer = Buffer.alloc(vector.length * 4);
  for (let i = 0; i < vector.length; i++) {
    buffer.writeFloatLE(vector[i], i * 4);
  }
  return buffer;
}

/**
 * Decode a Buffer back to a float32 array.
 */
function decodeEmbedding(buffer: Buffer): number[] {
  const length = buffer.length / 4;
  const result = new Array(length);
  for (let i = 0; i < length; i++) {
    result[i] = buffer.readFloatLE(i * 4);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Database open / initialization
// ---------------------------------------------------------------------------

/**
 * Resolve the database file path.
 */
function resolveDbPath(): string {
  const homeDir = process.env.HOME || '.';
  const dir = path.join(homeDir, '.pi', 'agent', 'extensions', 'session-log-indexer');
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, 'index.db');
}

/**
 * Open (or create) the SQLite database, initialize schema.
 */
export function openSessionDb(dbPath?: string): SessionDb {
  const path = dbPath ?? resolveDbPath();

  const db = new Database(path);

  // Enable WAL mode for better concurrent read performance
  db.pragma('journal_mode = WAL');

  // Enable FTS5
  db.pragma('enable_fts5 = ON');

  // Load sqlite-vec extension
  sqliteVec.load(db);

  // Create schema
  db.exec(SCHEMA_SQL);

  return db;
}

// ---------------------------------------------------------------------------
// CRUD Operations
// ---------------------------------------------------------------------------

/**
 * Check if a session is already indexed.
 */
export function isIndexed(db: SessionDb, sessionId: string): boolean {
  const row = db.prepare('SELECT 1 FROM session_chunks WHERE session_id = ? LIMIT 1').get(sessionId) as { id: string } | undefined;
  return row !== undefined;
}

/**
 * Get list of all indexed sessions with metadata.
 */
export function getIndexedSessions(db: SessionDb): IndexedSessionInfo[] {
  const rows = db.prepare(`
    SELECT
      session_id,
      session_ts,
      cwd,
      model,
      COUNT(*) as chunks
    FROM session_chunks
    GROUP BY session_id
    ORDER BY session_ts DESC
  `).all() as Array<{
    session_id: string;
    session_ts: string;
    cwd: string | null;
    model: string | null;
    chunks: number;
  }>;

  return rows.map(r => ({
    sessionId: r.session_id,
    timestamp: r.session_ts,
    cwd: r.cwd,
    model: r.model,
    chunks: r.chunks,
  }));
}

/**
 * Bulk insert chunks and their vectors.
 * Returns the number of rows inserted.
 */
export function insertChunks(
  db: SessionDb,
  chunks: ChunkRow[],
  vectors: Array<{ chunk_id: string; embedding: number[] }>,
): number {
  const insertChunk = db.prepare(`
    INSERT OR IGNORE INTO session_chunks (id, session_id, session_ts, cwd, model, role, turn_index, chunk_index, token_count, text, fts_id)
    VALUES (@id, @session_id, @session_ts, @cwd, @model, @role, @turn_index, @chunk_index, @token_count, @text, @fts_id)
  `);

  const insertVector = db.prepare(`
    INSERT OR REPLACE INTO session_chunk_vectors (chunk_id, embedding, model)
    VALUES (@chunk_id, @embedding, @model)
  `);

  const insertMany = db.transaction((chunksToInsert: ChunkRow[], vectorsToInsert: typeof vectors) => {
    let count = 0;
    for (const chunk of chunksToInsert) {
      const result = insertChunk.run(chunk);
      if (result.changes > 0) count++;
    }

    for (const v of vectorsToInsert) {
      const embeddingBuffer = encodeEmbedding(v.embedding);
      insertVector.run({
        chunk_id: v.chunk_id,
        embedding: embeddingBuffer,
        model: 'nomic-embed-text-v1.5',
      });
    }

    return count;
  });

  return insertMany(chunks, vectors);
}

/**
 * Delete all chunks for a session.
 */
export function deleteSessionChunks(db: SessionDb, sessionId: string): number {
  const deleteChunks = db.prepare('DELETE FROM session_chunks WHERE session_id = ?');
  const deleteVectors = db.prepare('DELETE FROM session_chunk_vectors WHERE chunk_id IN (SELECT id FROM session_chunks WHERE session_id = ?)');

  deleteVectors.run(sessionId);
  const result = deleteChunks.run(sessionId);
  return result.changes;
}

// ---------------------------------------------------------------------------
// Search Operations
// ---------------------------------------------------------------------------

/**
 * Build WHERE clause and params from search filters.
 */
function buildFilterQuery(filters: SearchFilters): { sql: string; params: unknown[] } {
  const conditions: string[] = [];
  const params: unknown[] = [];

  if (filters.cwdFilter) {
    conditions.push('cwd = ?');
    params.push(filters.cwdFilter);
  }

  if (filters.dateFrom) {
    conditions.push('session_ts >= ?');
    params.push(filters.dateFrom);
  }

  if (filters.dateTo) {
    conditions.push('session_ts <= ?');
    params.push(filters.dateTo);
  }

  if (filters.role) {
    conditions.push('role = ?');
    params.push(filters.role);
  }

  const where = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
  return { sql: where, params };
}

/**
 * Vector search using sqlite-vec.
 */
export function searchByEmbedding(
  db: SessionDb,
  embedding: number[],
  maxResults: number,
  filters: SearchFilters,
): SearchResult[] {
  const embeddingBuffer = encodeEmbedding(embedding);
  const { sql: whereClause, params: whereParams } = buildFilterQuery(filters);

  // sqlite-vec kNN search with distance metric
  const sql = `
    SELECT
      sc.id,
      sc.session_id,
      sc.session_ts,
      sc.cwd,
      sc.model,
      sc.role,
      sc.turn_index,
      sc.chunk_index,
      sc.token_count,
      sc.text,
      sc.fts_id,
      vec_distance_cosine(sv.embedding, ?) AS distance
    FROM session_chunks sc
    JOIN session_chunk_vectors sv ON sv.chunk_id = sc.id
    ${whereClause}
    ORDER BY distance ASC
    LIMIT ?
  `;

  const rows = db.prepare(sql).all(embeddingBuffer, maxResults) as Array<{
    id: string;
    session_id: string;
    session_ts: string;
    cwd: string | null;
    model: string | null;
    role: string;
    turn_index: number;
    chunk_index: number;
    token_count: number;
    text: string;
    fts_id: number;
    distance: number;
  }>;

  // Convert distance to similarity score (1 - distance)
  return rows.map(r => ({
    sessionId: r.session_id,
    sessionTimestamp: r.session_ts,
    cwd: r.cwd,
    model: r.model,
    role: r.role,
    turnIndex: r.turn_index,
    chunkIndex: r.chunk_index,
    tokenCount: r.token_count,
    text: r.text,
    score: Math.round((1 - r.distance) * 100) / 100,
  }));
}

/**
 * FTS5 text search using BM25.
 */
export function searchByText(
  db: SessionDb,
  query: string,
  maxResults: number,
  filters: SearchFilters,
): SearchResult[] {
  const { sql: whereClause, params: whereParams } = buildFilterQuery(filters);

  // FTS5 search with BM25 ranking
  const sql = `
    SELECT
      sc.id,
      sc.session_id,
      sc.session_ts,
      sc.cwd,
      sc.model,
      sc.role,
      sc.turn_index,
      sc.chunk_index,
      sc.token_count,
      sc.text,
      sc.fts_id,
      bm25(session_chunks_fts) AS rank
    FROM session_chunks_fts fts
    JOIN session_chunks sc ON sc.fts_id = fts.rowid
    ${whereClause}
    ORDER BY rank ASC
    LIMIT ?
  `;

  const rows = db.prepare(sql).all(...whereParams, maxResults) as Array<{
    id: string;
    session_id: string;
    session_ts: string;
    cwd: string | null;
    model: string | null;
    role: string;
    turn_index: number;
    chunk_index: number;
    token_count: number;
    text: string;
    fts_id: number;
    rank: number;
  }>;

  // Convert BM25 rank to a similarity score (lower rank = better, invert for display)
  return rows.map(r => ({
    sessionId: r.session_id,
    sessionTimestamp: r.session_ts,
    cwd: r.cwd,
    model: r.model,
    role: r.role,
    turnIndex: r.turn_index,
    chunkIndex: r.chunk_index,
    tokenCount: r.token_count,
    text: r.text,
    score: Math.max(0, Math.round((1 / (1 + r.rank)) * 100) / 100),
  }));
}

/**
 * Hybrid search: try vector first, fall back to FTS5.
 */
export function searchHybrid(
  db: SessionDb,
  embedding: number[] | null,
  query: string,
  maxResults: number,
  filters: SearchFilters,
): { method: 'vector' | 'fts5'; results: SearchResult[] } {
  if (embedding) {
    const results = searchByEmbedding(db, embedding, maxResults, filters);
    if (results.length > 0) {
      return { method: 'vector', results };
    }
  }

  // Fallback to FTS5
  const results = searchByText(db, query, maxResults, filters);
  return { method: 'fts5', results };
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/**
 * Get aggregate index statistics.
 */
export function getIndexStats(db: SessionDb): IndexStats {
  const totalSessions = db.prepare('SELECT COUNT(DISTINCT session_id) as count FROM session_chunks').get() as { count: number };
  const totalChunks = db.prepare('SELECT COUNT(*) as count FROM session_chunks').get() as { count: number };
  const totalTokens = db.prepare('SELECT COALESCE(SUM(token_count), 0) as total FROM session_chunks').get() as { total: number };
  const indexedSessions = getIndexedSessions(db);

  return {
    totalSessions: totalSessions.count,
    totalChunks: totalChunks.count,
    totalTokens: totalTokens.total,
    indexedSessions,
  };
}

/**
 * Check if any sessions are indexed at all.
 */
export function hasIndexedSessions(db: SessionDb): boolean {
  const row = db.prepare('SELECT 1 FROM session_chunks LIMIT 1').get() as { id: string } | undefined;
  return row !== undefined;
}
