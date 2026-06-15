/**
 * Session Log Indexer — search.ts Unit Tests
 *
 * Tests the embedding API and search logic with mocked fetch.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  getEmbedding,
  batchGetEmbedding,
  searchSessions,
  DEFAULT_SEARCH_CONFIG,
  type SearchConfig,
} from '../search.ts';
import type { Database } from 'better-sqlite3';
import type { SearchFilters, SearchResult } from '../db.ts';

// ---------------------------------------------------------------------------
// Mock global fetch
// ---------------------------------------------------------------------------

const MOCK_EMBEDDING = Array(768).fill(0).map(() => Math.random() - 0.5);

function mockLmStudioResponse(texts: string[], embedding = MOCK_EMBEDDING) {
  return {
    ok: true,
    json: async () => ({
      data: texts.map(() => ({ embedding })),
    }),
  };
}

// ---------------------------------------------------------------------------
// getEmbedding tests
// ---------------------------------------------------------------------------

describe('getEmbedding', () => {
  const config: SearchConfig = {
    ...DEFAULT_SEARCH_CONFIG,
    embeddingEndpoint: 'http://10.1.1.145:1234/v1/embeddings',
  };

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('calls LMStudio with correct payload and nomic task prefix', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['hello']));

    const result = await getEmbedding('hello', config, 1000);

    expect(fetchMock).toHaveBeenCalledWith(
      config.embeddingEndpoint,
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.embeddingModel,
          input: 'search_document: hello',
        }),
      }),
    );
    expect(result).toEqual(MOCK_EMBEDDING);
  });

  it('parses response correctly', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const customEmbedding = [0.1, 0.2, 0.3];
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test'], customEmbedding));

    const result = await getEmbedding('test', config, 1000);

    expect(result).toEqual(customEmbedding);
  });

  it('handles LMStudio unavailable', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockRejectedValue(new Error('Connection refused'));

    const result = await getEmbedding('test', config, 1000);

    expect(result).toBeNull();
  });

  it('handles non-OK response', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
    });

    const result = await getEmbedding('test', config, 1000);

    expect(result).toBeNull();
  });

  it('handles missing embedding in response', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [] }),
    });

    const result = await getEmbedding('test', config, 1000);

    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// batchGetEmbedding tests
// ---------------------------------------------------------------------------

describe('batchGetEmbedding', () => {
  const config: SearchConfig = {
    ...DEFAULT_SEARCH_CONFIG,
    embeddingEndpoint: 'http://10.1.1.145:1234/v1/embeddings',
  };

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends batched request for multiple texts', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(
      mockLmStudioResponse(['text1', 'text2', 'text3']),
    );

    const result = await batchGetEmbedding(['text1', 'text2', 'text3'], config, 10000);

    expect(fetchMock).toHaveBeenCalledWith(
      config.embeddingEndpoint,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          model: config.embeddingModel,
          input: ['search_document: text1', 'search_document: text2', 'search_document: text3'],
        }),
      }),
    );
    expect(result).toHaveLength(3);
    expect(result).toEqual([MOCK_EMBEDDING, MOCK_EMBEDDING, MOCK_EMBEDDING]);
  });

  it('returns empty array for empty input', async () => {
    const result = await batchGetEmbedding([], config, 10000);
    expect(result).toEqual([]);
  });

  it('returns null embeddings on failure', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockRejectedValue(new Error('Connection refused'));

    const result = await batchGetEmbedding(['text1', 'text2'], config, 10000);

    expect(result).toEqual([null, null]);
  });

  it('handles non-OK response', async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue({ ok: false, status: 500 });

    const result = await batchGetEmbedding(['text1'], config, 10000);

    expect(result).toEqual([null]);
  });
});

// ---------------------------------------------------------------------------
// searchSessions tests
// ---------------------------------------------------------------------------

describe('searchSessions', () => {
  // Create a mock database with the required methods
  function createMockDb(): Database.Database {
    return {
      exec: vi.fn(),
      prepare: vi.fn(() => ({
        all: vi.fn(() => []),
        get: vi.fn(() => undefined),
        run: vi.fn(() => ({ changes: 0 })),
      })),
      pragma: vi.fn(),
      close: vi.fn(),
      changeCounter: 0,
      dump: vi.fn(),
    } as unknown as Database.Database;
  }

  const config: SearchConfig = {
    ...DEFAULT_SEARCH_CONFIG,
    embeddingEndpoint: 'http://10.1.1.145:1234/v1/embeddings',
  };

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns empty results for empty query', async () => {
    const mockDb = createMockDb();
    const result = await searchSessions(mockDb, '', 10, {}, config);

    expect(result.method).toBe('fts5');
    expect(result.results).toEqual([]);
  });

  it('returns empty results when no indexed sessions', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    const result = await searchSessions(mockDb, 'test query', 10, {}, config);

    expect(result.results).toEqual([]);
  });

  it('returns vector results when embedding available and matches', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    // Mock the vector search to return results
    const prepareMock = vi.fn(() => ({
      all: vi.fn(() => [
        {
          id: 'chunk-1',
          session_id: 'abc-123',
          session_ts: '2026-06-14T11:00:00.000Z',
          cwd: '/home/pi',
          model: 'qwen/v1',
          role: 'assistant',
          turn_index: 0,
          chunk_index: 0,
          token_count: 50,
          text: 'hello world',
          fts_id: 1,
          distance: 0.13,
        },
      ]),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const result = await searchSessions(mockDb, 'test query', 10, {}, config);

    expect(result.method).toBe('vector');
    expect(result.results).toHaveLength(1);
    expect(result.results[0].sessionId).toBe('abc-123');
    expect(result.results[0].score).toBe(0.87);
  });

  it('falls back to FTS5 when vector search returns nothing', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    // Mock vector search to return empty, then FTS5 to return results
    let callCount = 0;
    const prepareMock = vi.fn(() => ({
      all: vi.fn(() => {
        callCount++;
        if (callCount === 1) return []; // Vector search returns nothing
        return [
          {
            id: 'chunk-1',
            session_id: 'abc-123',
            session_ts: '2026-06-14T11:00:00.000Z',
            cwd: '/home/pi',
            model: 'qwen/v1',
            role: 'assistant',
            turn_index: 0,
            chunk_index: 0,
            token_count: 50,
            text: 'hello world',
            fts_id: 1,
            rank: 0.5,
          },
        ];
      }),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const result = await searchSessions(mockDb, 'test query', 10, {}, config);

    expect(result.method).toBe('fts5');
    expect(result.results).toHaveLength(1);
  });

  it('applies role filter', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    const prepareMock = vi.fn(() => ({
      all: vi.fn(() => [
        {
          id: 'chunk-1',
          session_id: 'abc-123',
          session_ts: '2026-06-14T11:00:00.000Z',
          cwd: '/home/pi',
          model: 'qwen/v1',
          role: 'user',
          turn_index: 0,
          chunk_index: 0,
          token_count: 50,
          text: 'hello',
          fts_id: 1,
          distance: 0.1,
        },
      ]),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const filters: SearchFilters = { role: 'user' };
    const result = await searchSessions(mockDb, 'test query', 10, filters, config);

    expect(result.results[0].role).toBe('user');
  });

  it('respects maxResults', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    const prepareMock = vi.fn(() => ({
      all: vi.fn((...args: unknown[]) => {
        // The last argument should be maxResults
        const limit = args[args.length - 1];
        return Array(Math.min(limit as number, 3)).fill(null).map((_, i) => ({
          id: `chunk-${i}`,
          session_id: 'abc-123',
          session_ts: '2026-06-14T11:00:00.000Z',
          cwd: '/home/pi',
          model: 'qwen/v1',
          role: 'assistant',
          turn_index: i,
          chunk_index: 0,
          token_count: 50,
          text: `result ${i}`,
          fts_id: i,
          distance: 0.1 * (i + 1),
        }));
      }),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const result = await searchSessions(mockDb, 'test query', 2, {}, config);

    expect(result.results.length).toBe(2);
  });

  it('handles LMStudio unavailable — falls back to FTS5', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockRejectedValue(new Error('Connection refused'));

    const prepareMock = vi.fn(() => ({
      all: vi.fn(() => [
        {
          id: 'chunk-1',
          session_id: 'abc-123',
          session_ts: '2026-06-14T11:00:00.000Z',
          cwd: '/home/pi',
          model: 'qwen/v1',
          role: 'assistant',
          turn_index: 0,
          chunk_index: 0,
          token_count: 50,
          text: 'hello world',
          fts_id: 1,
          rank: 0.5,
        },
      ]),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const result = await searchSessions(mockDb, 'test query', 10, {}, config);

    expect(result.method).toBe('fts5');
    expect(result.results).toHaveLength(1);
  });

  it('applies cwdFilter', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    const prepareMock = vi.fn(() => ({
      all: vi.fn(() => [
        {
          id: 'chunk-1',
          session_id: 'abc-123',
          session_ts: '2026-06-14T11:00:00.000Z',
          cwd: '/home/pi/project',
          model: 'qwen/v1',
          role: 'assistant',
          turn_index: 0,
          chunk_index: 0,
          token_count: 50,
          text: 'hello',
          fts_id: 1,
          distance: 0.1,
        },
      ]),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const filters: SearchFilters = { cwdFilter: '/home/pi/project' };
    const result = await searchSessions(mockDb, 'test query', 10, filters, config);

    expect(result.results[0].cwd).toBe('/home/pi/project');
  });

  it('applies date range filter', async () => {
    const mockDb = createMockDb();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(mockLmStudioResponse(['test query']));

    const prepareMock = vi.fn(() => ({
      all: vi.fn(() => [
        {
          id: 'chunk-1',
          session_id: 'abc-123',
          session_ts: '2026-06-14T11:00:00.000Z',
          cwd: '/home/pi',
          model: 'qwen/v1',
          role: 'assistant',
          turn_index: 0,
          chunk_index: 0,
          token_count: 50,
          text: 'hello',
          fts_id: 1,
          distance: 0.1,
        },
      ]),
    }));
    (mockDb.prepare as ReturnType<typeof vi.fn>).mockImplementation(prepareMock);

    const filters: SearchFilters = {
      dateFrom: '2026-06-14T00:00:00.000Z',
      dateTo: '2026-06-14T23:59:59.999Z',
    };
    const result = await searchSessions(mockDb, 'test query', 10, filters, config);

    expect(result.results[0].sessionTimestamp).toBe('2026-06-14T11:00:00.000Z');
  });
});
