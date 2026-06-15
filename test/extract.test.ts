/**
 * Session Log Indexer — extract.ts Unit Tests
 *
 * Tests pure functions: parseSession, getSessionMetadata,
 * extractForeground, chunkTurn, extractChunks, scanErrors,
 * scanToolUsage, scanSessionStats.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  parseSession,
  getSessionMetadata,
  extractForeground,
  chunkTurn,
  extractChunks,
  scanErrors,
  scanToolUsage,
  scanSessionStats,
  type SessionRecord,
  type Turn,
  type Chunk,
} from '../extract.ts';

// ---------------------------------------------------------------------------
// Helpers — create temporary .jsonl files
// ---------------------------------------------------------------------------

function writeTempSession(lines: string[]): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'session-test-'));
  const filepath = path.join(dir, 'test.jsonl');
  fs.writeFileSync(filepath, lines.join('\n') + '\n');
  return filepath;
}

function mockSessionRecord(id: string, timestamp: string, cwd: string): SessionRecord {
  return {
    type: 'session',
    version: 1,
    id,
    timestamp,
    cwd,
  };
}

function mockModelChange(model: string | null): SessionRecord {
  return {
    type: 'model_change',
    model,
  };
}

function mockMessage(role: string, contentBlocks: unknown[]): SessionRecord {
  return {
    type: 'message',
    id: `msg-${role}`,
    timestamp: '2026-06-14T12:00:00.000Z',
    message: {
      role,
      content: contentBlocks,
      model: null,
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      stopReason: 'stop',
      timestamp: Date.now(),
    },
  };
}

function mockToolCall(toolName: string, toolCallId: string): SessionRecord {
  return {
    type: 'tool_call',
    toolName,
    toolCallId,
    input: { command: 'echo hello' },
  };
}

function mockToolResult(toolCallId: string, isError: boolean, error?: string): SessionRecord {
  return {
    type: 'tool_result',
    toolCallId,
    isError,
    content: error ? { type: 'error', error } : { type: 'text', text: 'done' },
  };
}

// ---------------------------------------------------------------------------
// parseSession tests
// ---------------------------------------------------------------------------

describe('parseSession', () => {
  it('reads valid .jsonl into records array', () => {
    const filepath = writeTempSession([
      JSON.stringify({ type: 'session', id: 'test-1' }),
      JSON.stringify({ type: 'message', role: 'user' }),
    ]);
    const records = parseSession(filepath);
    expect(records).toHaveLength(2);
    expect(records[0].type).toBe('session');
    expect(records[1].type).toBe('message');
  });

  it('skips blank lines', () => {
    const filepath = writeTempSession([
      '',
      JSON.stringify({ type: 'session', id: 'test-1' }),
      '',
      '',
      JSON.stringify({ type: 'message', role: 'user' }),
      '',
    ]);
    const records = parseSession(filepath);
    expect(records).toHaveLength(2);
  });

  it('handles malformed JSON gracefully', () => {
    const filepath = writeTempSession([
      JSON.stringify({ type: 'session', id: 'test-1' }),
      '{ not valid json }',
      JSON.stringify({ type: 'message', role: 'user' }),
    ]);
    const records = parseSession(filepath);
    expect(records).toHaveLength(2);
    expect(records[0].type).toBe('session');
    expect(records[1].type).toBe('message');
  });
});

// ---------------------------------------------------------------------------
// getSessionMetadata tests
// ---------------------------------------------------------------------------

describe('getSessionMetadata', () => {
  it('extracts all fields from session and model_change records', () => {
    const records: SessionRecord[] = [
      { type: 'session', id: 'abc-123', timestamp: '2026-06-14T11:00:00.000Z', cwd: '/home/pi/project' },
      { type: 'model_change', model: 'qwen/qwen3.6-35b' },
    ];
    const meta = getSessionMetadata(records);
    expect(meta.sessionId).toBe('abc-123');
    expect(meta.timestamp).toBe('2026-06-14T11:00:00.000Z');
    expect(meta.cwd).toBe('/home/pi/project');
    expect(meta.model).toBe('qwen/qwen3.6-35b');
  });

  it('handles null model', () => {
    const records: SessionRecord[] = [
      { type: 'session', id: 'abc-123', timestamp: '2026-06-14T11:00:00.000Z', cwd: '/home/pi' },
      { type: 'model_change', model: null },
    ];
    const meta = getSessionMetadata(records);
    expect(meta.model).toBeNull();
  });

  it('uses last model_change when multiple exist', () => {
    const records: SessionRecord[] = [
      { type: 'session', id: 'abc-123', timestamp: '2026-06-14T11:00:00.000Z', cwd: '/home/pi' },
      { type: 'model_change', model: 'model/v1' },
      { type: 'model_change', model: 'model/v2' },
    ];
    const meta = getSessionMetadata(records);
    expect(meta.model).toBe('model/v2');
  });

  it('returns empty model when no model_change records', () => {
    const records: SessionRecord[] = [
      { type: 'session', id: 'abc-123', timestamp: '2026-06-14T11:00:00.000Z', cwd: '/home/pi' },
    ];
    const meta = getSessionMetadata(records);
    expect(meta.model).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// extractForeground tests
// ---------------------------------------------------------------------------

describe('extractForeground', () => {
  it('includes user text', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(1);
    expect(turns[0].role).toBe('user');
    expect(turns[0].text).toBe('hello');
  });

  it('includes assistant text', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('assistant', [{ type: 'text', text: 'hi there' }]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(1);
    expect(turns[0].role).toBe('assistant');
    expect(turns[0].text).toBe('hi there');
  });

  it('excludes thinking blocks', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('assistant', [
        { type: 'thinking', thinking: 'let me think...' },
        { type: 'text', text: 'the answer is 42' },
      ]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(1);
    expect(turns[0].text).toBe('the answer is 42');
    expect(turns[0].text).not.toContain('let me think');
  });

  it('excludes toolCall blocks', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('assistant', [
        { type: 'toolCall', name: 'bash', arguments: { command: 'ls' } },
        { type: 'text', text: 'done listing' },
      ]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(1);
    expect(turns[0].text).toBe('done listing');
    expect(turns[0].text).not.toContain('bash');
  });

  it('excludes toolResult blocks', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('assistant', [
        { type: 'text', text: 'here is the result' },
        { type: 'toolResult', content: { text: 'output' } },
      ]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(1);
    expect(turns[0].text).toBe('here is the result');
    expect(turns[0].text).not.toContain('output');
  });

  it('skips non-message records', () => {
    const records: SessionRecord[] = [
      { type: 'session', id: 's1', timestamp: '2026-06-14T11:00:00.000Z', cwd: '/home/pi' },
      { type: 'model_change', model: 'qwen/v1' },
      { type: 'thinking_level_change', level: 0.5 },
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
      mockMessage('assistant', [{ type: 'text', text: 'hi' }]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(2);
    expect(turns[0].role).toBe('user');
    expect(turns[1].role).toBe('assistant');
  });

  it('captures multiple text blocks in one message', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('assistant', [
        { type: 'text', text: 'first part' },
        { type: 'thinking', thinking: '...' },
        { type: 'text', text: 'second part' },
      ]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(1);
    expect(turns[0].text).toBe('first part\nsecond part');
  });

  it('skips empty text turns', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('assistant', [{ type: 'thinking', thinking: 'only thinking' }]),
    ];
    const turns = extractForeground(records);
    expect(turns).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// chunkTurn tests
// ---------------------------------------------------------------------------

describe('chunkTurn', () => {
  const meta = {
    sessionId: 'abc-123',
    timestamp: '2026-06-14T11:00:00.000Z',
    cwd: '/home/pi',
    model: 'qwen/v1',
  };

  it('single chunk under 512 tokens', () => {
    const chunks = chunkTurn(meta, 0, 'user', 'hello world');
    expect(chunks).toHaveLength(1);
    expect(chunks[0].chunkIndex).toBe(0);
    expect(chunks[0].text).toBe('hello world');
    expect(chunks[0].tokenCount).toBeGreaterThan(0);
  });

  it('multiple chunks over 512 tokens', () => {
    // ~2048 chars ≈ 512 tokens
    const longText = 'a'.repeat(2200);
    const chunks = chunkTurn(meta, 0, 'user', longText);
    expect(chunks.length).toBeGreaterThan(1);
    // Verify sequential chunkIndex
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].chunkIndex).toBe(i);
    }
    // Verify no overlap — concatenated text should equal original
    const reconstructed = chunks.map(c => c.text).join('');
    expect(reconstructed.length).toBe(longText.length);
  });

  it('exact 512 token boundary — one chunk', () => {
    // ~2048 chars ≈ 512 tokens
    const boundaryText = 'x'.repeat(2048);
    const chunks = chunkTurn(meta, 0, 'user', boundaryText);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].chunkIndex).toBe(0);
  });

  it('very large turn produces many sub-chunks', () => {
    // ~21K tokens ≈ 84K chars
    const hugeText = 'b'.repeat(85000);
    const chunks = chunkTurn(meta, 0, 'user', hugeText);
    expect(chunks.length).toBeGreaterThan(30);
    expect(chunks[0].chunkIndex).toBe(0);
    expect(chunks[chunks.length - 1].chunkIndex).toBe(chunks.length - 1);
  });

  it('preserves role metadata through chunks', () => {
    const longText = 'c'.repeat(3000);
    const chunks = chunkTurn(meta, 2, 'assistant', longText);
    for (const chunk of chunks) {
      expect(chunk.role).toBe('assistant');
      expect(chunk.turnIndex).toBe(2);
      expect(chunk.sessionId).toBe('abc-123');
      expect(chunk.cwd).toBe('/home/pi');
      expect(chunk.model).toBe('qwen/v1');
    }
  });

  it('empty text produces no chunks', () => {
    const chunks = chunkTurn(meta, 0, 'user', '');
    expect(chunks).toHaveLength(0);
  });

  it('whitespace-only text produces no chunks', () => {
    const chunks = chunkTurn(meta, 0, 'user', '   \n  \n  ');
    expect(chunks).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// extractChunks tests
// ---------------------------------------------------------------------------

describe('extractChunks', () => {
  it('full pipeline: session → turns → chunks', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockModelChange('qwen/v1'),
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
      mockMessage('assistant', [{ type: 'text', text: 'hi back' }]),
    ];
    const chunks = extractChunks(records);
    expect(chunks).toHaveLength(2);
    expect(chunks[0].role).toBe('user');
    expect(chunks[0].text).toBe('hello');
    expect(chunks[0].turnIndex).toBe(0);
    expect(chunks[0].chunkIndex).toBe(0);
    expect(chunks[1].role).toBe('assistant');
    expect(chunks[1].text).toBe('hi back');
    expect(chunks[1].turnIndex).toBe(1);
  });

  it('chunks carry session metadata', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('my-session', '2026-06-14T11:00:00.000Z', '/proj'),
      mockModelChange('model/x'),
      mockMessage('user', [{ type: 'text', text: 'test' }]),
    ];
    const chunks = extractChunks(records);
    expect(chunks[0].sessionId).toBe('my-session');
    expect(chunks[0].sessionTimestamp).toBe('2026-06-14T11:00:00.000Z');
    expect(chunks[0].cwd).toBe('/proj');
    expect(chunks[0].model).toBe('model/x');
  });
});

// ---------------------------------------------------------------------------
// scanErrors tests
// ---------------------------------------------------------------------------

describe('scanErrors', () => {
  it('finds tool errors', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockToolCall('bash', 'call-1'),
      mockToolResult('call-1', true, 'Command failed with exit code 1'),
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
    ];
    const errors = scanErrors(records);
    expect(errors).toHaveLength(1);
    expect(errors[0].toolName).toBe('bash');
    expect(errors[0].toolCallId).toBe('call-1');
    expect(errors[0].errorMessage).toBe('Command failed with exit code 1');
  });

  it('skips non-error tool results', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockToolCall('bash', 'call-1'),
      mockToolResult('call-1', false),
    ];
    const errors = scanErrors(records);
    expect(errors).toHaveLength(0);
  });

  it('finds multiple errors', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockToolCall('bash', 'call-1'),
      mockToolResult('call-1', true, 'error 1'),
      mockToolCall('read', 'call-2'),
      mockToolResult('call-2', true, 'error 2'),
      mockToolCall('read', 'call-3'),
      mockToolResult('call-3', false),
    ];
    const errors = scanErrors(records);
    expect(errors).toHaveLength(2);
    expect(errors[0].toolCallId).toBe('call-1');
    expect(errors[1].toolCallId).toBe('call-2');
  });

  it('returns empty array when no errors', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
    ];
    const errors = scanErrors(records);
    expect(errors).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// scanToolUsage tests
// ---------------------------------------------------------------------------

describe('scanToolUsage', () => {
  it('counts tool invocations by name', () => {
    const records: SessionRecord[] = [
      mockToolCall('bash', 'c1'),
      mockToolCall('read', 'c2'),
      mockToolCall('bash', 'c3'),
      mockToolCall('bash', 'c4'),
      mockToolCall('read', 'c5'),
    ];
    const usages = scanToolUsage(records);
    expect(usages).toHaveLength(2);
    expect(usages[0].toolName).toBe('bash');
    expect(usages[0].count).toBe(3);
    expect(usages[1].toolName).toBe('read');
    expect(usages[1].count).toBe(2);
  });

  it('returns empty array when no tool calls', () => {
    const records: SessionRecord[] = [
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
    ];
    const usages = scanToolUsage(records);
    expect(usages).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// scanSessionStats tests
// ---------------------------------------------------------------------------

describe('scanSessionStats', () => {
  it('computes aggregate session statistics', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockModelChange('qwen/v1'),
      mockMessage('user', [{ type: 'text', text: 'hello' }]),
      mockMessage('assistant', [{ type: 'text', text: 'hi' }]),
    ];
    const stats = scanSessionStats(records);
    expect(stats.sessionId).toBe('s1');
    expect(stats.timestamp).toBe('2026-06-14T11:00:00.000Z');
    expect(stats.cwd).toBe('/home/pi');
    expect(stats.turnCount).toBe(2);
    expect(stats.errorCount).toBe(0);
    expect(stats.models).toContain('qwen/v1');
  });

  it('counts errors in stats', () => {
    const records: SessionRecord[] = [
      mockSessionRecord('s1', '2026-06-14T11:00:00.000Z', '/home/pi'),
      mockToolCall('bash', 'c1'),
      mockToolResult('c1', true, 'error'),
      mockToolCall('bash', 'c2'),
      mockToolResult('c2', true, 'error'),
    ];
    const stats = scanSessionStats(records);
    expect(stats.errorCount).toBe(2);
  });
});
