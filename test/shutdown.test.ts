/**
 * Session Log Indexer — shutdown hook tests
 *
 * Tests indexSessionOnShutdown behavior:
 * - No-op when session is already indexed
 * - No-op when session file has no valid header
 * - Successful indexing of a new session
 * - Graceful degradation on missing embedding API
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import { openSessionDb, isIndexed, deleteSessionChunks } from '../db.ts';
import { indexSessionOnShutdown } from '../index.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let tmpDir: string;
let dbPath: string;

function writeTempSession(sessionId: string, timestamp: string, cwd: string, lines: string[]): string {
  const header = JSON.stringify({ type: 'session', id: sessionId, timestamp, cwd });
  const allLines = [header, ...lines];
  const filepath = path.join(tmpDir, `${sessionId}.jsonl`);
  fs.writeFileSync(filepath, allLines.join('\n') + '\n');
  return filepath;
}

/**
 * Build a properly-formatted message record (matches real .jsonl structure).
 */
function msgRecord(role: string, text: string): string {
  return JSON.stringify({
    type: 'message',
    id: crypto.randomUUID().slice(0, 8),
    parentId: null,
    timestamp: '2026-07-04T12:00:00.000Z',
    message: {
      role,
      content: [{ type: 'text', text }],
      timestamp: Date.now(),
    },
  });
}

function createDb(): Database.Database {
  const db = openSessionDb(dbPath);
  return db;
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'shutdown-test-'));
  dbPath = path.join(tmpDir, 'test.db');
});

afterEach(() => {
  // Clean up temp files
  try {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  } catch { /* ignore */ }
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('indexSessionOnShutdown', () => {
  it('should skip indexing when session is already indexed', async () => {
    const db = createDb();
    const sessionId = 'test-already-indexed-session-id';
    const filepath = writeTempSession(
      sessionId,
      '2026-07-04T12:00:00.000Z',
      '/tmp/test',
      [
        msgRecord('user', 'Hello'),
        msgRecord('assistant', 'Hi there'),
      ],
    );

    // Pre-index the session by inserting a dummy chunk
    const insert = db.prepare(`
      INSERT OR IGNORE INTO session_chunks (id, session_id, session_ts, cwd, model, role, turn_index, chunk_index, token_count, text, fts_id)
      VALUES (@id, @session_id, @session_ts, @cwd, @model, @role, @turn_index, @chunk_index, @token_count, @text, @fts_id)
    `);
    insert.run({
      id: crypto.randomUUID(),
      session_id: sessionId,
      session_ts: '2026-07-04T12:00:00.000Z',
      cwd: '/tmp/test',
      model: null,
      role: 'user',
      turn_index: 0,
      chunk_index: 0,
      token_count: 1,
      text: 'dummy',
      fts_id: 1,
    });

    expect(isIndexed(db, sessionId)).toBe(true);

    // Run the shutdown indexer — should skip
    await indexSessionOnShutdown(filepath, dbPath);

    // Still exactly 1 chunk (the dummy)
    const count = db.prepare('SELECT COUNT(*) as c FROM session_chunks WHERE session_id = ?').get(sessionId) as { c: number };
    expect(count.c).toBe(1);
    db.close();
  });

  it('should skip indexing when session file has no valid header', async () => {
    const filepath = path.join(tmpDir, 'bad-header.jsonl');
    fs.writeFileSync(filepath, 'not valid json\n');

    // Should not throw
    await indexSessionOnShutdown(filepath);
  });

  it('should skip indexing when session file does not exist', async () => {
    // Should not throw
    await indexSessionOnShutdown('/nonexistent/path/session.jsonl');
  });

  it('should index a new session (graceful degradation without embedding API)', async () => {
    const sessionId = 'test-new-session-to-index';
    const filepath = writeTempSession(
      sessionId,
      '2026-07-04T12:00:00.000Z',
      '/tmp/test',
      [
        msgRecord('user', 'What is TypeScript?'),
        msgRecord('assistant', 'TypeScript is a typed superset of JavaScript.'),
      ],
    );

    const db = createDb();
    expect(isIndexed(db, sessionId)).toBe(false);

    // Run the shutdown indexer
    await indexSessionOnShutdown(filepath, dbPath);

    // Session should now be indexed (chunks stored even without embeddings)
    expect(isIndexed(db, sessionId)).toBe(true);

    const count = db.prepare('SELECT COUNT(*) as c FROM session_chunks WHERE session_id = ?').get(sessionId) as { c: number };
    expect(count.c).toBeGreaterThanOrEqual(1);

    // Verify chunk content
    const chunks = db.prepare('SELECT text FROM session_chunks WHERE session_id = ?').all(sessionId) as Array<{ text: string }>;
    const allText = chunks.map(c => c.text).join(' ');
    expect(allText).toContain('TypeScript');

    db.close();
  });

  it('should not duplicate chunks when run twice', async () => {
    const sessionId = 'test-dedup-session';
    const filepath = writeTempSession(
      sessionId,
      '2026-07-04T12:00:00.000Z',
      '/tmp/test',
      [
        msgRecord('user', 'Hello world'),
        msgRecord('assistant', 'Hello to you too!'),
      ],
    );

    // First indexing
    await indexSessionOnShutdown(filepath, dbPath);

    const db = createDb();
    const firstCount = db.prepare('SELECT COUNT(*) as c FROM session_chunks WHERE session_id = ?').get(sessionId) as { c: number };

    // Second indexing — should be a no-op (already indexed)
    await indexSessionOnShutdown(filepath, dbPath);

    const secondCount = db.prepare('SELECT COUNT(*) as c FROM session_chunks WHERE session_id = ?').get(sessionId) as { c: number };

    expect(firstCount.c).toBe(secondCount.c);
    db.close();
  });
});
