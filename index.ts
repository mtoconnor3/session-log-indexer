/**
 * Session Log Indexer — Entry Point
 *
 * Registers 7 tools with Pi's extension system:
 *
 *   session_extract      — Extract foreground conversation from sessions
 *   session_index        — Index sessions (extract → chunk → embed → store)
 *   session_index_status — Check what's already indexed
 *   session_search       — Semantic search across indexed conversations
 *   session_errors       — Find failed tool calls
 *   session_stats        — Aggregate statistics
 *   session_list         — Browse sessions with metadata
 *
 * Also registers a session_shutdown hook that auto-indexes the current
 * session when the user quits (Ctrl+D, Ctrl+C, SIGTERM).
 */

import fs from 'fs';
import { randomUUID } from 'crypto';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { openSessionDb, isIndexed } from './db.ts';
import { registerTools } from './tools.ts';

export default async function (pi: ExtensionAPI): Promise<void> {
  // Open (or create) the SQLite database
  const db = openSessionDb();

  console.log('[session-log-indexer] Session log indexer initialized');

  // Register tools — agent-triggered
  registerTools(db, pi);

  // Auto-index on quit: fire-and-forget, doesn't block shutdown
  pi.on('session_shutdown', async (event, ctx) => {
    if (event.reason !== 'quit') return;

    const sessionFile = ctx.sessionManager.getSessionFile();
    if (!sessionFile) return;

    // Delay to ensure the session file fd is fully flushed / closed
    setTimeout(() => {
      void indexSessionOnShutdown(sessionFile);
    }, 500);
  });
}

/**
 * Index a session file on shutdown.  Fire-and-forget — errors are logged
 * but never thrown so they can't block the shutdown sequence.
 */
async function indexSessionOnShutdown(sessionFile: string): Promise<void> {
  try {
    const db = openSessionDb();

    // Read session ID from header
    const firstLine = fs.readFileSync(sessionFile, 'utf-8').split('\n')[0];
    const header = JSON.parse(firstLine) as { id?: string };
    const sessionId = header.id;
    if (!sessionId) { db.close(); return; }

    if (isIndexed(db, sessionId)) {
      console.log(`[session-log-indexer] Session ${sessionId.slice(0, 8)} already indexed, skipping`);
      db.close();
      return;
    }

    // Dynamic import to avoid blocking the shutdown handler
    const extract = await import('./extract.ts');
    const search = await import('./search.ts');

    const records = extract.parseSession(sessionFile);
    const chunks = extract.extractChunks(records);

    if (chunks.length === 0) { db.close(); return; }

    const maxFtsRow = db.prepare('SELECT COALESCE(MAX(fts_id), -1) as max_fts FROM session_chunks').get();
    const baseFtsId = (maxFtsRow as { max_fts: number }).max_fts;

    const rows = chunks.map((c, i) => ({
      id: randomUUID(),
      session_id: c.sessionId,
      session_ts: c.sessionTimestamp,
      cwd: c.cwd,
      model: c.model,
      role: c.role,
      turn_index: c.turnIndex,
      chunk_index: c.chunkIndex,
      token_count: c.tokenCount,
      text: c.text,
      fts_id: baseFtsId + i + 1,
    }));

    console.log(`[session-log-indexer] Indexing ${rows.length} chunks for session ${sessionId.slice(0, 8)}...`);

    search.indexSessions(db, rows, search.DEFAULT_SEARCH_CONFIG, { force: false })
      .then((result) => {
        console.log(
          `[session-log-indexer] Indexed ${result.indexed} chunks (${result.errors} errors) for session ${sessionId.slice(0, 8)}`,
        );
        db.close();
      })
      .catch((err: Error) => {
        console.warn(`[session-log-indexer] Failed to index session: ${err.message}`);
        db.close();
      });
  } catch (err) {
    console.warn(`[session-log-indexer] Shutdown index error: ${(err as Error).message}`);
  }
}
