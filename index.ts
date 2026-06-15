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
 * No hooks, no commands, no side effects on load.
 * Tools are called on-demand by the agent.
 */

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { openSessionDb } from './db.ts';
import { registerTools } from './tools.ts';

export default async function (pi: ExtensionAPI): Promise<void> {
  // Open (or create) the SQLite database
  const db = openSessionDb();

  console.log('[session-log-indexer] Session log indexer initialized');

  // Register tools — agent-triggered only, no hooks
  registerTools(db, pi);
}
