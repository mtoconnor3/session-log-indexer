/**
 * Session Log Indexer — Session Parsing (Pure Functions)
 *
 * Parses .jsonl session files, extracts foreground conversation,
 * and chunks turns at ~512 tokens. Zero side effects.
 *
 * Record types found in session files:
 *   - session:           metadata (id, timestamp, cwd)
 *   - model_change:      model switch event
 *   - thinking_level_change: thinking level change event
 *   - message:           user/assistant conversation with content blocks
 *   - tool_call:         tool invocation (for error scanning)
 *   - tool_result:       tool response (for error scanning)
 *
 * Content blocks within messages:
 *   - text:              foreground conversation (included)
 *   - thinking:          reasoning content (filtered out)
 *   - toolCall:          tool invocation details (filtered out)
 *   - toolResult:        tool response details (filtered out)
 */

import fs from 'fs';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A parsed JSON record from a session file. */
export interface SessionRecord {
  type: string;
  [key: string]: unknown;
}

/** Metadata extracted from a session's header records. */
export interface SessionMetadata {
  sessionId: string;
  timestamp: string;
  cwd: string;
  model: string | null;
}

/** A single turn of foreground conversation. */
export interface Turn {
  role: string;
  text: string;
}

/** A chunked piece of a turn. */
export interface Chunk {
  sessionId: string;
  sessionTimestamp: string;
  cwd: string;
  model: string | null;
  role: string;
  turnIndex: number;
  chunkIndex: number;
  tokenCount: number;
  text: string;
}

/** An error found in a tool result. */
export interface SessionError {
  sessionId: string;
  timestamp: string;
  cwd: string;
  toolName: string;
  toolCallId: string;
  errorMessage: string;
}

/** Tool usage statistics from a session. */
export interface ToolUsage {
  toolName: string;
  count: number;
}

/** Aggregate statistics for a session. */
export interface SessionStats {
  sessionId: string;
  timestamp: string;
  cwd: string;
  turnCount: number;
  tokenCountIn: number;
  tokenCountOut: number;
  errorCount: number;
  models: string[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHUNK_TOKEN_LIMIT = 512;
const TOKEN_CHAR_RATIO = 4;

// ---------------------------------------------------------------------------
// parseSession — Read .jsonl → records array
// ---------------------------------------------------------------------------

/**
 * Read a session .jsonl file and parse each line into a record.
 * Skips blank lines and malformed JSON gracefully.
 */
export function parseSession(filepath: string): SessionRecord[] {
  const content = fs.readFileSync(filepath, 'utf-8');
  const lines = content.split('\n');
  const records: SessionRecord[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    try {
      const record = JSON.parse(trimmed) as SessionRecord;
      records.push(record);
    } catch {
      // Skip malformed JSON lines — they're not fatal
      continue;
    }
  }

  return records;
}

// ---------------------------------------------------------------------------
// getSessionMetadata — Extract session header info
// ---------------------------------------------------------------------------

/**
 * Extract metadata from session records.
 * The session record provides id, timestamp, cwd.
 * The last model_change record provides the model.
 */
export function getSessionMetadata(records: SessionRecord[]): SessionMetadata {
  let sessionId = '';
  let timestamp = '';
  let cwd = '';
  let model: string | null = null;

  for (const record of records) {
    if (record.type === 'session') {
      sessionId = (record.id as string) || '';
      timestamp = (record.timestamp as string) || '';
      cwd = (record.cwd as string) || '';
    } else if (record.type === 'model_change') {
      const m = record.model;
      if (m !== null && m !== undefined) {
        model = String(m);
      }
    }
  }

  return { sessionId, timestamp, cwd, model };
}

// ---------------------------------------------------------------------------
// extractForeground — Extract (role, text) pairs per turn
// ---------------------------------------------------------------------------

/**
 * Extract foreground conversation from session records.
 * - Includes user and assistant messages
 * - Extracts only 'text' content blocks (filters thinking, toolCall, toolResult)
 * - Skips non-message records
 */
export function extractForeground(
  records: SessionRecord[],
): Turn[] {
  const turns: Turn[] = [];

  for (const record of records) {
    if (record.type !== 'message') continue;

    const message = record.message as Record<string, unknown> | undefined;
    if (!message) continue;

    const role = message.role as string | undefined;
    if (role !== 'user' && role !== 'assistant') continue;

    const content = message.content as Array<Record<string, unknown>> | undefined;
    if (!content || !Array.isArray(content)) continue;

    // Collect only 'text' block content
    const textParts: string[] = [];
    for (const block of content) {
      if (block.type === 'text' && typeof block.text === 'string') {
        textParts.push(block.text);
      }
      // Skip thinking, toolCall, toolResult blocks
    }

    const text = textParts.join('\n');
    if (text.trim()) {
      turns.push({ role, text });
    }
  }

  return turns;
}

// ---------------------------------------------------------------------------
// chunkTurn — Split a turn into ~512-token chunks
// ---------------------------------------------------------------------------

/**
 * Estimate token count from character count.
 * Rough approximation: 1 token ≈ 4 characters.
 */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / TOKEN_CHAR_RATIO));
}

/**
 * Split a turn's text into chunks at the token budget boundary.
 * Never splits in the middle of a word.
 */
function splitByTokens(text: string, limit: number): string[] {
  const tokens = estimateTokens(text);

  if (tokens <= limit) {
    return [text];
  }

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > 0) {
    const remainingTokens = estimateTokens(remaining);

    if (remainingTokens <= limit) {
      chunks.push(remaining);
      break;
    }

    // Target character count for this chunk
    const targetChars = Math.floor(limit * TOKEN_CHAR_RATIO);
    let splitPoint = Math.min(targetChars, remaining.length);

    // Find the last whitespace before splitPoint to avoid splitting words
    if (splitPoint < remaining.length) {
      const spaceIdx = remaining.lastIndexOf(' ', splitPoint);
      if (spaceIdx > 0) {
        splitPoint = spaceIdx;
      }
    }

    // If no space found, force split at target
    if (splitPoint <= 0) {
      splitPoint = Math.max(1, Math.floor(targetChars * 0.8));
    }

    chunks.push(remaining.slice(0, splitPoint).trimEnd());
    remaining = remaining.slice(splitPoint).trimStart();
  }

  return chunks;
}

/**
 * Chunk a single turn into pieces respecting the token budget.
 * Returns an array of Chunk objects with metadata.
 */
export function chunkTurn(
  metadata: SessionMetadata,
  turnIndex: number,
  role: string,
  text: string,
): Chunk[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  const parts = splitByTokens(trimmed, CHUNK_TOKEN_LIMIT);

  return parts.map((part, chunkIndex) => ({
    sessionId: metadata.sessionId,
    sessionTimestamp: metadata.timestamp,
    cwd: metadata.cwd,
    model: metadata.model,
    role,
    turnIndex,
    chunkIndex,
    tokenCount: estimateTokens(part),
    text: part,
  }));
}

// ---------------------------------------------------------------------------
// extractChunks — Full pipeline: records → chunks
// ---------------------------------------------------------------------------

/**
 * Full extraction pipeline: parse records → foreground turns → chunks.
 */
export function extractChunks(
  records: SessionRecord[],
): Chunk[] {
  const metadata = getSessionMetadata(records);
  const turns = extractForeground(records);
  const allChunks: Chunk[] = [];

  for (let turnIndex = 0; turnIndex < turns.length; turnIndex++) {
    const turn = turns[turnIndex];
    const chunks = chunkTurn(metadata, turnIndex, turn.role, turn.text);
    allChunks.push(...chunks);
  }

  return allChunks;
}

// ---------------------------------------------------------------------------
// scanErrors — Find failed tool calls in session records
// ---------------------------------------------------------------------------

/**
 * Scan session records for tool_result entries with isError: true.
 * Used by session_errors tool (on-demand path, not indexed).
 */
export function scanErrors(records: SessionRecord[]): SessionError[] {
  const errors: SessionError[] = [];
  const metadata = getSessionMetadata(records);

  // Build a map of toolCallId → toolName from tool_call records
  const callNames = new Map<string, string>();
  for (const record of records) {
    if (record.type === 'tool_call' && record.toolCallId && record.toolName) {
      callNames.set(String(record.toolCallId), String(record.toolName));
    }
  }

  for (const record of records) {
    if (record.type !== 'tool_result') continue;

    const isError = record.isError as boolean | undefined;
    if (!isError) continue;

    const content = record.content as Record<string, unknown> | undefined;
    const errorMessage =
      typeof content?.error === 'string'
        ? content.error
        : typeof content?.text === 'string'
          ? content.text
          : JSON.stringify(content ?? '');

    const toolCallId = record.toolCallId as string || '';
    const toolName = callNames.get(toolCallId) || 'unknown';

    errors.push({
      sessionId: metadata.sessionId,
      timestamp: metadata.timestamp,
      cwd: metadata.cwd,
      toolName,
      toolCallId,
      errorMessage: errorMessage.substring(0, 2000),
    });
  }

  return errors;
}

// ---------------------------------------------------------------------------
// scanToolUsage — Count tool invocations
// ---------------------------------------------------------------------------

/**
 * Scan session records for tool_call entries and count by tool name.
 */
export function scanToolUsage(records: SessionRecord[]): ToolUsage[] {
  const counts = new Map<string, number>();

  for (const record of records) {
    if (record.type !== 'tool_call') continue;
    const toolName = record.toolName as string;
    if (toolName) {
      counts.set(toolName, (counts.get(toolName) || 0) + 1);
    }
  }

  return Array.from(counts.entries())
    .map(([toolName, count]) => ({ toolName, count }))
    .sort((a, b) => b.count - a.count);
}

// ---------------------------------------------------------------------------
// scanSessionStats — Aggregate statistics for a session
// ---------------------------------------------------------------------------

/**
 * Compute aggregate stats for a session from its records.
 */
export function scanSessionStats(records: SessionRecord[]): SessionStats {
  const metadata = getSessionMetadata(records);
  const turns = extractForeground(records);
  const errors = scanErrors(records);
  const toolUsages = scanToolUsage(records);

  // Collect models from model_change records
  const modelSet = new Set<string>();
  for (const record of records) {
    if (record.type === 'model_change' && record.model) {
      modelSet.add(String(record.model));
    }
    // Also collect models from message usage
    if (record.type === 'message') {
      const msg = record.message as Record<string, unknown> | undefined;
      if (msg?.model) {
        modelSet.add(String(msg.model));
      }
    }
  }

  // Count tokens from message usage fields
  let tokenCountIn = 0;
  let tokenCountOut = 0;
  for (const record of records) {
    if (record.type === 'message') {
      const msg = record.message as Record<string, unknown> | undefined;
      const usage = msg?.usage as Record<string, unknown> | undefined;
      if (usage) {
        tokenCountIn += (usage.input as number) || 0;
        tokenCountOut += (usage.output as number) || 0;
      }
    }
  }

  return {
    sessionId: metadata.sessionId,
    timestamp: metadata.timestamp,
    cwd: metadata.cwd,
    turnCount: turns.length,
    tokenCountIn,
    tokenCountOut,
    errorCount: errors.length,
    models: Array.from(modelSet),
  };
}
