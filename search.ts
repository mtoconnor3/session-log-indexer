/**
 * Session Log Indexer — Embedding + Search Layer
 *
 * Handles LMStudio embedding API integration, hybrid search (FTS5 + vectors),
 * and search orchestration.
 */

import type { SessionDb } from './db.ts';
import type {
  SearchFilters,
  SearchResult,
  IndexStats,
} from './db.ts';
import {
  searchHybrid,
  searchByEmbedding,
  searchByText,
  getIndexStats,
  isIndexed,
  insertChunks,
  deleteSessionChunks,
  hasIndexedSessions,
  openSessionDb,
  type ChunkRow,
} from './db.ts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface SearchConfig {
  embeddingEndpoint: string;
  embeddingModel: string;
  maxResults: number;
  embeddingTimeoutMs: number;
}

export const DEFAULT_SEARCH_CONFIG: SearchConfig = {
  embeddingEndpoint: 'http://10.1.1.145:1234/v1/embeddings',
  embeddingModel: 'nomic-embed-text-v1.5',
  maxResults: 10,
  embeddingTimeoutMs: 10000,
};

// ---------------------------------------------------------------------------
// Embedding API
// ---------------------------------------------------------------------------

/**
 * Call LMStudio to generate an embedding for a single text.
 * Returns null if the endpoint is unavailable.
 */
export async function getEmbedding(
  text: string,
  config: SearchConfig = DEFAULT_SEARCH_CONFIG,
  timeoutMs: number = 10000,
): Promise<number[] | null> {
  try {
    // Apply nomic task prefix
    const prefixed = `search_document: ${text}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(config.embeddingEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: prefixed,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.warn(`[session-log-indexer] Embedding API returned ${response.status}`);
      return null;
    }

    const data = await response.json() as { data: Array<{ embedding: number[] }> };
    return data.data?.[0]?.embedding ?? null;
  } catch (err) {
    console.warn(`[session-log-indexer] Embedding request failed: ${(err as Error).message}`);
    return null;
  }
}

/**
 * Batch-embed multiple texts in a single API call.
 * Returns an array of embeddings (null for failures).
 */
export async function batchGetEmbedding(
  texts: string[],
  config: SearchConfig = DEFAULT_SEARCH_CONFIG,
  timeoutMs: number = 30000,
): Promise<Array<number[] | null>> {
  if (texts.length === 0) return [];

  try {
    const prefixed = texts.map(t => `search_document: ${t}`);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(config.embeddingEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: prefixed,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.warn(`[session-log-indexer] Batch embedding API returned ${response.status}`);
      return texts.map(() => null);
    }

    const data = await response.json() as { data: Array<{ embedding: number[] }> };
    return data.data?.map(d => d.embedding) ?? texts.map(() => null);
  } catch (err) {
    console.warn(`[session-log-indexer] Batch embedding failed: ${(err as Error).message}`);
    return texts.map(() => null);
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/**
 * Search indexed sessions using hybrid search (vector → FTS5 fallback).
 */
export async function searchSessions(
  db: SessionDb,
  query: string,
  maxResults: number,
  filters: SearchFilters,
  config: SearchConfig = DEFAULT_SEARCH_CONFIG,
): Promise<{ method: 'vector' | 'fts5'; results: SearchResult[] }> {
  if (!query.trim()) {
    return { method: 'fts5', results: [] };
  }

  // Get embedding for the query
  const embedding = await getEmbedding(query, config, config.embeddingTimeoutMs);

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
// Indexing
// ---------------------------------------------------------------------------

/**
 * Index sessions: extract → chunk → embed → store.
 */
export interface IndexConfig {
  force?: boolean;
}

export async function indexSessions(
  db: SessionDb,
  chunks: ChunkRow[],
  config: SearchConfig = DEFAULT_SEARCH_CONFIG,
  indexConfig: IndexConfig = {},
): Promise<{
  success: boolean;
  indexed: number;
  skipped: number;
  chunks: number;
  errors: number;
}> {
  const totalChunks = chunks.length;
  if (totalChunks === 0) {
    return { success: true, indexed: 0, skipped: 0, chunks: 0, errors: 0 };
  }

  const sessionId = chunks[0].session_id;

  // Check if already indexed
  if (!indexConfig.force && isIndexed(db, sessionId)) {
    return { success: true, indexed: 0, skipped: totalChunks, chunks: totalChunks, errors: 0 };
  }

  // Remove existing index for this session (if force mode)
  if (indexConfig.force) {
    deleteSessionChunks(db, sessionId);
  }

  // Batch embed all chunks
  const texts = chunks.map(c => c.text);
  const embeddings = await batchGetEmbedding(texts, config, 60000);

  // Build vector rows (only for successful embeddings)
  const vectors: Array<{ chunk_id: string; embedding: number[] }> = [];
  let errors = 0;

  for (let i = 0; i < chunks.length; i++) {
    const emb = embeddings[i];
    if (emb !== null) {
      vectors.push({ chunk_id: chunks[i].id, embedding: emb });
    } else {
      errors++;
    }
  }

  // Insert chunks and vectors
  const inserted = insertChunks(db, chunks, vectors);

  return {
    success: true,
    indexed: inserted,
    skipped: totalChunks - inserted,
    chunks: totalChunks,
    errors,
  };
}
