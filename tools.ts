/**
 * Session Log Indexer — Tool Implementations
 *
 * Thin wrappers that parse parameters, call extract/db/search,
 * and format output as { content: [{ type: "text", text: JSON.stringify(result) }] }.
 */

import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import type Database from 'better-sqlite3';
import {
  parseSession,
  extractChunks,
  getSessionMetadata,
  scanErrors,
  scanToolUsage,
  scanSessionStats,
  type SessionRecord,
  type Chunk,
  type SessionError,
  type ToolUsage,
  type SessionStats,
} from './extract.ts';
import type {
  SearchFilters,
  SearchResult,
  IndexStats,
  IndexedSessionInfo,
  ChunkRow,
} from './db.ts';
import {
  openSessionDb,
  getIndexedSessions,
  getIndexStats,
  isIndexed,
  searchHybrid,
  type SessionDb,
} from './db.ts';
import {
  searchSessions,
  indexSessions,
  getEmbedding,
  DEFAULT_SEARCH_CONFIG,
  type SearchConfig,
  type IndexConfig,
} from './search.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SESSIONS_DIR = path.join(process.env.HOME || '.', '.pi', 'agent', 'sessions');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a list of session file paths from filters (Layer 1: filesystem, no file reads).
 * Uses lexicographic filename sorting = chronological sorting.
 */
function listSessionFiles(filters: {
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
  sessionId?: string;
  limit?: number;
}): string[] {
  const files: string[] = [];

  if (!fs.existsSync(SESSIONS_DIR)) {
    return files;
  }

  // If sessionId is specified, try to find the exact file
  if (filters.sessionId) {
    const pattern = `${filters.sessionId}.jsonl`;
    const found = findFileBySessionId(pattern);
    if (found) return [found];
    return [];
  }

  // Walk session directories
  const directories = fs.readdirSync(SESSIONS_DIR).filter(d => {
    const dirPath = path.join(SESSIONS_DIR, d);
    return fs.statSync(dirPath).isDirectory();
  });

  for (const dir of directories) {
    const dirPath = path.join(SESSIONS_DIR, dir);
    const sessionFiles = fs.readdirSync(dirPath)
      .filter(f => f.endsWith('.jsonl'))
      .sort(); // Lexicographic sort = chronological

    for (const file of sessionFiles) {
      const filepath = path.join(dirPath, file);

      // Apply date range filter using filename prefix (YYYY-MM-DDTHH-MM-SS-mmmZ_...)
      if (filters.dateFrom || filters.dateTo) {
        const fileName = file.replace('.jsonl', '');
        const fileDate = fileName.split('_')[0]?.replace(/-/g, ':').replace('T', 'T');
        // Filename format: 2026-06-14T11-24-24-260Z_UUID
        // Convert to comparable: 2026-06-14T11:24:24.260Z
        const comparableDate = fileName
          .replace('T', 'T')
          .replace(/(\d{2})-(\d{2})-(\d{2})-(\d{3})Z/, 'T$1:$2:$3.$4Z');

        if (filters.dateFrom && comparableDate < filters.dateFrom) continue;
        if (filters.dateTo && comparableDate > filters.dateTo) continue;
      }

      // Apply cwdFilter by reading session header (lightweight, just first line)
      if (filters.cwdFilter) {
        try {
          const firstLine = fs.readFileSync(filepath, 'utf-8').split('\n')[0];
          const header = JSON.parse(firstLine) as { cwd?: string };
          if (header.cwd !== filters.cwdFilter) continue;
        } catch {
          continue;
        }
      }

      files.push(filepath);

      if (filters.limit && files.length >= filters.limit) {
        return files;
      }
    }
  }

  return files;
}

/**
 * Find a file by session ID pattern.
 */
function findFileBySessionId(pattern: string): string | null {
  if (!fs.existsSync(SESSIONS_DIR)) return null;

  const directories = fs.readdirSync(SESSIONS_DIR).filter(d => {
    const dirPath = path.join(SESSIONS_DIR, d);
    return fs.statSync(dirPath).isDirectory();
  });

  for (const dir of directories) {
    const dirPath = path.join(SESSIONS_DIR, dir);
    const files = fs.readdirSync(dirPath);
    const match = files.find(f => f.includes(pattern));
    if (match) return path.join(dirPath, match);
  }

  return null;
}

/**
 * Check if a session file is already indexed.
 */
function isSessionIndexed(db: SessionDb, filepath: string): boolean {
  // Read session header to get sessionId
  try {
    const firstLine = fs.readFileSync(filepath, 'utf-8').split('\n')[0];
    const header = JSON.parse(firstLine) as { id?: string };
    if (!header.id) return false;
    return isIndexed(db, header.id);
  } catch {
    return false;
  }
}

/**
 * Format tool output.
 */
function formatOutput(result: unknown): { content: Array<{ type: 'text'; text: string }>; details: unknown } {
  return {
    content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
    details: result,
  };
}

// ---------------------------------------------------------------------------
// Tool Implementations
// ---------------------------------------------------------------------------

/**
 * 5.1 session_extract — Extract foreground conversation from sessions.
 */
export function sessionExtract(
  db: SessionDb,
  params: Record<string, unknown>,
): ReturnType<typeof formatOutput> {
  const sessionId = params.sessionId as string | undefined;
  const cwdFilter = params.cwdFilter as string | undefined;
  const dateFrom = params.dateFrom as string | undefined;
  const dateTo = params.dateTo as string | undefined;
  const limit = (params.limit as number) ?? 1;

  const files = listSessionFiles({ cwdFilter, dateFrom, dateTo, sessionId, limit });

  if (sessionId && files.length === 0) {
    return formatOutput({ success: false, message: 'Session not found' });
  }

  const allChunks: Chunk[] = [];
  for (const filepath of files) {
    const records = parseSession(filepath);
    const chunks = extractChunks(records);
    allChunks.push(...chunks);
  }

  return formatOutput({
    success: true,
    sessionsExtracted: files.length,
    chunks: allChunks,
  });
}

/**
 * 5.2 session_index — Index sessions: extract → chunk → embed → store.
 */
export async function sessionIndex(
  db: SessionDb,
  params: Record<string, unknown>,
): Promise<ReturnType<typeof formatOutput>> {
  const sessionId = params.sessionId as string | undefined;
  const cwdFilter = params.cwdFilter as string | undefined;
  const dateFrom = params.dateFrom as string | undefined;
  const dateTo = params.dateTo as string | undefined;
  const force = params.force as boolean | undefined;

  const files = listSessionFiles({ cwdFilter, dateFrom, dateTo, sessionId });

  // Filter out already-indexed sessions (unless force)
  const filesToIndex = files.filter(f => !force && !isSessionIndexed(db, f));

  let totalIndexed = 0;
  let totalSkipped = 0;
  let totalChunks = 0;
  let totalErrors = 0;

  for (const filepath of filesToIndex) {
    const records = parseSession(filepath);
    const chunks = extractChunks(records);

    if (chunks.length === 0) continue;

    // Convert Chunk → ChunkRow for DB
    const rows: ChunkRow[] = chunks.map((c, i) => ({
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
      fts_id: i + 1, // Simple sequential fts_id per session
    }));

    const result = await indexSessions(db, rows, DEFAULT_SEARCH_CONFIG, { force: !!force });
    totalIndexed += result.indexed;
    totalSkipped += result.skipped;
    totalChunks += result.chunks;
    totalErrors += result.errors;
  }

  // Also count skipped files
  const skippedFiles = files.length - filesToIndex.length;
  totalSkipped += skippedFiles * 100; // Approximate — actual chunk count varies

  return formatOutput({
    success: true,
    indexed: totalIndexed,
    skipped: totalSkipped,
    chunks: totalChunks,
    errors: totalErrors,
  });
}

/**
 * 5.3 session_index_status — Check what's already indexed.
 */
export function sessionIndexStatus(
  db: SessionDb,
  _params: Record<string, unknown>,
): ReturnType<typeof formatOutput> {
  const stats = getIndexStats(db);
  return formatOutput({
    success: true,
    totalSessions: stats.totalSessions,
    totalChunks: stats.totalChunks,
    totalTokens: stats.totalTokens,
    indexedSessions: stats.indexedSessions,
  });
}

/**
 * 5.4 session_search — Semantic search across indexed conversations.
 */
export async function sessionSearch(
  db: SessionDb,
  params: Record<string, unknown>,
): Promise<ReturnType<typeof formatOutput>> {
  const query = params.query as string | undefined;
  const cwdFilter = params.cwdFilter as string | undefined;
  const dateFrom = params.dateFrom as string | undefined;
  const dateTo = params.dateTo as string | undefined;
  const role = params.role as string | undefined;
  const maxResults = (params.maxResults as number) ?? 10;

  if (!query || !query.trim()) {
    return formatOutput({ success: false, message: 'Query is required' });
  }

  const filters: SearchFilters = { cwdFilter, dateFrom, dateTo, role };

  const result = await searchSessions(db, query, maxResults, filters, DEFAULT_SEARCH_CONFIG);

  return formatOutput({
    success: true,
    query,
    method: result.method,
    results: result.results,
  });
}

/**
 * 5.5 session_errors — Find failed tool calls across session logs.
 */
export function sessionErrors(
  _db: SessionDb,
  params: Record<string, unknown>,
): ReturnType<typeof formatOutput> {
  const cwdFilter = params.cwdFilter as string | undefined;
  const dateFrom = params.dateFrom as string | undefined;
  const dateTo = params.dateTo as string | undefined;
  const toolName = params.toolName as string | undefined;
  const maxResults = (params.maxResults as number) ?? 20;

  const files = listSessionFiles({ cwdFilter, dateFrom, dateTo });

  const allErrors: SessionError[] = [];

  for (const filepath of files) {
    const records = parseSession(filepath);
    const errors = scanErrors(records);

    if (toolName) {
      allErrors.push(...errors.filter(e => e.toolName === toolName));
    } else {
      allErrors.push(...errors);
    }
  }

  // Sort by timestamp (newest first) and limit
  allErrors.sort((a, b) => (b.timestamp > a.timestamp ? 1 : -1));
  const limited = allErrors.slice(0, maxResults);

  return formatOutput({
    success: true,
    totalCount: allErrors.length,
    errors: limited,
  });
}

/**
 * 5.6 session_stats — Aggregate statistics across session logs.
 */
export function sessionStats(
  _db: SessionDb,
  params: Record<string, unknown>,
): ReturnType<typeof formatOutput> {
  const cwdFilter = params.cwdFilter as string | undefined;
  const dateFrom = params.dateFrom as string | undefined;
  const dateTo = params.dateTo as string | undefined;

  const files = listSessionFiles({ cwdFilter, dateFrom, dateTo });

  let totalSessions = 0;
  let totalTurns = 0;
  let totalErrors = 0;
  let totalTokensIn = 0;
  let totalTokensOut = 0;
  const toolUsageMap = new Map<string, number>();
  const sessionsByProject = new Map<string, number>();

  for (const filepath of files) {
    const records = parseSession(filepath);
    const stats = scanSessionStats(records);
    const usages = scanToolUsage(records);

    totalSessions++;
    totalTurns += stats.turnCount;
    totalErrors += stats.errorCount;
    totalTokensIn += stats.tokenCountIn;
    totalTokensOut += stats.tokenCountOut;

    sessionsByProject.set(stats.cwd ?? 'unknown', (sessionsByProject.get(stats.cwd ?? 'unknown') || 0) + 1);

    for (const u of usages) {
      toolUsageMap.set(u.toolName, (toolUsageMap.get(u.toolName) || 0) + u.count);
    }
  }

  const toolUsage = Array.from(toolUsageMap.entries())
    .sort((a, b) => b[1] - a[1])
    .reduce((acc, [toolName, count]) => {
      acc[toolName] = count;
      return acc;
    }, {} as Record<string, number>);

  const sessionsByProjectObj = Object.fromEntries(
    Array.from(sessionsByProject.entries()).sort((a, b) => b[1] - a[1]),
  );

  return formatOutput({
    success: true,
    totalSessions,
    totalTurns,
    totalErrors,
    totalTokensIn,
    totalTokensOut,
    toolUsage,
    sessionsByProject: sessionsByProjectObj,
  });
}

/**
 * 5.7 session_list — Browse sessions with metadata.
 */
export function sessionList(
  db: SessionDb,
  params: Record<string, unknown>,
): ReturnType<typeof formatOutput> {
  const cwdFilter = params.cwdFilter as string | undefined;
  const dateFrom = params.dateFrom as string | undefined;
  const dateTo = params.dateTo as string | undefined;
  const limit = (params.limit as number) ?? 50;
  const offset = (params.offset as number) ?? 0;
  const sortBy = params.sortBy as string | undefined;

  const files = listSessionFiles({ cwdFilter, dateFrom, dateTo });

  // Parse metadata for each session
  const sessions: Array<{
    sessionId: string;
    timestamp: string;
    cwd: string;
    file: string;
    size: number;
    turnCount: number;
    tokenCountIn: number;
    tokenCountOut: number;
    errorCount: number;
    models: string[];
    isIndexed: boolean;
  }> = [];

  for (const filepath of files) {
    const records = parseSession(filepath);
    const stats = scanSessionStats(records);

    // Read session header for model info
    const models = stats.models;

    const fileName = path.basename(filepath);
    const fileSize = fs.statSync(filepath).size;

    sessions.push({
      sessionId: stats.sessionId,
      timestamp: stats.timestamp,
      cwd: stats.cwd,
      file: fileName,
      size: fileSize,
      turnCount: stats.turnCount,
      tokenCountIn: stats.tokenCountIn,
      tokenCountOut: stats.tokenCountOut,
      errorCount: stats.errorCount,
      models,
      isIndexed: isIndexed(db, stats.sessionId),
    });
  }

  // Sort if requested
  if (sortBy === 'tokensIn') {
    sessions.sort((a, b) => b.tokenCountIn - a.tokenCountIn);
  } else if (sortBy === 'turnCount') {
    sessions.sort((a, b) => b.turnCount - a.turnCount);
  } else {
    // Default: by timestamp (newest first)
    sessions.sort((a, b) => (b.timestamp > a.timestamp ? 1 : -1));
  }

  // Paginate
  const total = sessions.length;
  const paginated = sessions.slice(offset, offset + limit);

  return formatOutput({
    success: true,
    total,
    sessions: paginated,
  });
}

// ---------------------------------------------------------------------------
// Tool Registration
// ---------------------------------------------------------------------------

/**
 * Register all 7 tools with Pi.
 */
export function registerTools(db: SessionDb, pi: ExtensionAPI): void {
  // session_extract
  pi.registerTool({
    name: 'session_extract',
    label: 'Extract Session',
    description: 'Extract foreground conversation (user/assistant text) from one or more session logs. Excludes thinking blocks, tool calls, and tool results.',
    parameters: {
      type: 'object',
      properties: {
        sessionId: { type: 'string', description: 'Specific session UUID to extract.' },
        cwdFilter: { type: 'string', description: 'Scope to a project directory.' },
        dateFrom: { type: 'string', description: 'Start date (YYYY-MM-DD or ISO format).' },
        dateTo: { type: 'string', description: 'End date (YYYY-MM-DD or ISO format).' },
        limit: { type: 'number', description: 'Max sessions to extract (default 1).' },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, unknown>) =>
      sessionExtract(db, params),
  });

  // session_index
  pi.registerTool({
    name: 'session_index',
    label: 'Index Session',
    description: 'Index sessions: extract foreground conversation, chunk, embed via nomic-embed-text-v1.5, and store in SQLite. Skips already-indexed sessions unless force=true.',
    parameters: {
      type: 'object',
      properties: {
        sessionId: { type: 'string', description: 'Specific session UUID to index.' },
        cwdFilter: { type: 'string', description: 'Scope to a project directory.' },
        dateFrom: { type: 'string', description: 'Start date (YYYY-MM-DD or ISO format).' },
        dateTo: { type: 'string', description: 'End date (YYYY-MM-DD or ISO format).' },
        force: { type: 'boolean', description: 'Re-index even if already in DB (default false).' },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, unknown>) =>
      sessionIndex(db, params),
  });

  // session_index_status
  pi.registerTool({
    name: 'session_index_status',
    label: 'Index Status',
    description: 'Check what sessions are already indexed: total sessions, chunks, tokens, and per-session breakdown.',
    parameters: { type: 'object', properties: {} },
    execute: async (_toolCallId: string, _params: Record<string, unknown>) =>
      sessionIndexStatus(db, _params),
  });

  // session_search
  pi.registerTool({
    name: 'session_search',
    label: 'Search Sessions',
    description: 'Semantic search across indexed conversations using vector embeddings (nomic-embed-text-v1.5). Falls back to FTS5 text search if embedding unavailable.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query text.' },
        cwdFilter: { type: 'string', description: 'Scope to a project directory.' },
        dateFrom: { type: 'string', description: 'Start date (YYYY-MM-DD or ISO format).' },
        dateTo: { type: 'string', description: 'End date (YYYY-MM-DD or ISO format).' },
        role: { type: 'string', description: "Filter by role: 'user' or 'assistant'." },
        maxResults: { type: 'number', description: 'Max results (default 10).' },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, unknown>) =>
      sessionSearch(db, params),
  });

  // session_errors
  pi.registerTool({
    name: 'session_errors',
    label: 'Session Errors',
    description: 'Find failed tool calls across session logs. Scans raw .jsonl files for tool_result records with isError=true.',
    parameters: {
      type: 'object',
      properties: {
        cwdFilter: { type: 'string', description: 'Scope to a project directory.' },
        dateFrom: { type: 'string', description: 'Start date (YYYY-MM-DD or ISO format).' },
        dateTo: { type: 'string', description: 'End date (YYYY-MM-DD or ISO format).' },
        toolName: { type: 'string', description: 'Filter by tool name.' },
        maxResults: { type: 'number', description: 'Max results (default 20).' },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, unknown>) =>
      sessionErrors(db, params),
  });

  // session_stats
  pi.registerTool({
    name: 'session_stats',
    label: 'Session Stats',
    description: 'Aggregate statistics across session logs: total sessions, turns, errors, tokens, tool usage, and sessions by project.',
    parameters: {
      type: 'object',
      properties: {
        cwdFilter: { type: 'string', description: 'Scope to a project directory.' },
        dateFrom: { type: 'string', description: 'Start date (YYYY-MM-DD or ISO format).' },
        dateTo: { type: 'string', description: 'End date (YYYY-MM-DD or ISO format).' },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, unknown>) =>
      sessionStats(db, params),
  });

  // session_list
  pi.registerTool({
    name: 'session_list',
    label: 'List Sessions',
    description: 'Browse sessions with metadata: turn count, token counts, error count, models used, and indexed status.',
    parameters: {
      type: 'object',
      properties: {
        cwdFilter: { type: 'string', description: 'Scope to a project directory.' },
        dateFrom: { type: 'string', description: 'Start date (YYYY-MM-DD or ISO format).' },
        dateTo: { type: 'string', description: 'End date (YYYY-MM-DD or ISO format).' },
        limit: { type: 'number', description: 'Max results (default 50).' },
        offset: { type: 'number', description: 'Pagination offset (default 0).' },
        sortBy: { type: 'string', description: 'Sort by: "timestamp" | "turnCount" | "tokensIn".' },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, unknown>) =>
      sessionList(db, params),
  });
}
