# Session Log Indexer — Implementation Plan

## 1. Overview

A Pi extension that indexes and searches previous Pi session logs. It parses `.jsonl` session files, extracts foreground conversation (user/assistant text, excluding thinking and tool blocks), chunks at ~512 tokens, embeds via `nomic-embed-text-v1.5`, and stores everything in a standalone SQLite database.

The extension exposes 7 tools. All tools accept `cwdFilter`, `dateFrom`, and `dateTo` parameters so the agent can scope queries by project directory and date range — leveraging the structured filename format (`YYYY-MM-DDTHH-MM-SS-mmmZ_UUID.jsonl`) for fast filesystem-level filtering.

## 2. Architecture

### Two-Layer Design

```
Layer 1: Filesystem Index (fast, zero-parse)
  - Walk session directories under ~/.pi/agent/sessions/
  - Parse filenames for timestamp + UUID
  - String-compare for date range filtering (lexicographic sort = chronological sort)
  - Decode directory name → cwd (read session header as authoritative source)
  - Cost: microseconds, no .jsonl reads

Layer 2: Content Parser (slower, opens files)
  - Opens .jsonl files from Layer 1 results
  - Extracts conversation, tokens, errors, tool usage
  - Cost: milliseconds per file
```

### File Structure

```
session-log-indexer/
├── index.ts          # Entry point — tool registration only
├── db.ts             # SQLite setup, schema, CRUD operations
├── search.ts         # Embedding API (nomic via LMStudio) + hybrid search
├── tools.ts          # Tool implementations
├── extract.ts        # Session parsing + foreground extraction + chunking
├── package.json      # Dependencies
├── tsconfig.json     # TypeScript config
└── test/
    ├── extract.test.ts   # extract.ts unit tests
    └── search.test.ts    # search.ts unit tests (mocked LMStudio)
```

### Dependencies

Same as kg-memory:
- `better-sqlite3` — SQLite driver
- `@photostructure/sqlite-vec` — vector search in SQLite

## 3. Database Schema

### `session_chunks` Table

```sql
CREATE TABLE session_chunks (
    id              TEXT PRIMARY KEY,    -- UUID v4
    session_id      TEXT NOT NULL,       -- UUID from session header
    session_ts      TEXT,                -- ISO timestamp
    cwd             TEXT,                -- project directory
    model           TEXT,                -- model used (from model_change)
    role            TEXT NOT NULL,       -- 'user' | 'assistant'
    turn_index      INTEGER NOT NULL,    -- position in conversation
    chunk_index     INTEGER NOT NULL,    -- 0 for single-chunk turns
    token_count     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    fts_id          INTEGER UNIQUE       -- rowid for FTS5
);
```

### `session_chunks_fts` — FTS5 Virtual Table

```sql
CREATE VIRTUAL TABLE session_chunks_fts USING fts5(
    text,
    content='session_chunks',
    content_rowid='fts_id'
);
```

### `session_chunk_vectors` — Vector Store

```sql
CREATE TABLE session_chunk_vectors (
    chunk_id    TEXT PRIMARY KEY REFERENCES session_chunks(id),
    embedding   BLOB NOT NULL,           -- 768-dim float32 array
    model       TEXT NOT NULL            -- 'nomic-embed-text-v1.5'
);
```

### Indexes

```sql
CREATE INDEX idx_chunks_session ON session_chunks(session_id);
CREATE INDEX idx_chunks_cwd ON session_chunks(cwd);
CREATE INDEX idx_chunks_ts ON session_chunks(session_ts);
```

### Triggers (FTS5 sync)

```sql
CREATE TRIGGER chunks_ai AFTER INSERT ON session_chunks BEGIN
    INSERT INTO session_chunks_fts(rowid, text)
    VALUES (new.fts_id, new.text);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON session_chunks BEGIN
    DELETE FROM session_chunks_fts WHERE rowid = old.fts_id;
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON session_chunks BEGIN
    UPDATE session_chunks_fts SET text = new.text WHERE rowid = old.fts_id;
END;
```

## 4. Chunking Strategy

### Primary Division: Turn Boundary

Each user↔assistant turn is a chunk boundary. We never split across turns.

### Token Budget: 512 Tokens

- If a turn's text ≤ 512 tokens → one chunk (`chunkIndex: 0`)
- If a turn's text > 512 tokens → sequential sub-chunks (`chunkIndex: 0, 1, 2, ...`)

### Token Estimation

Rough estimate: `charCount / 4`. This is approximate but good enough for chunking decisions. The actual token count is stored per chunk for reference.

### Content Filtered Out

- `thinking` blocks (reasoning content)
- `toolCall` blocks (structural, not conversational)
- `toolResult` blocks (system automation)
- Non-message records (`session`, `model_change`, `thinking_level_change`)

### Edge Case: Oversized Turns

One observed user turn was 21,314 tokens (~85K chars). At 512 tokens/chunk, this produces ~42 sub-chunks. The `sessionId`, `turnIndex`, and `chunkIndex` metadata lets the agent reconstruct which chunks belong to the same turn.

## 5. Tool Specifications

### 5.1 `session_extract`

Extract foreground conversation from one or more sessions.

**Input:**
```typescript
{
  sessionId?: string;      // specific session UUID
  cwdFilter?: string;      // scope to project directory
  dateFrom?: string;       // "2026-06-10" or "2026-06-10T11:00:00"
  dateTo?: string;
  limit?: number;          // max sessions to extract (default 1)
}
```

**Behavior:**
1. Build file list from `cwdFilter` + `dateFrom`/`dateTo` (Layer 1: filesystem, no file reads)
2. If `sessionId` provided, find that specific file
3. Open matching `.jsonl` files (up to `limit`)
4. Extract foreground conversation: (role, text) pairs per turn
5. Chunk each turn at 512 tokens
6. Return chunks with metadata

**Output:**
```json
{
  "success": true,
  "sessionsExtracted": 1,
  "chunks": [
    {
      "sessionId": "019ec5e0-...",
      "sessionTimestamp": "2026-06-14T11:24:24.260Z",
      "cwd": "/home/pi",
      "model": "qwen/qwen3.6-35b-a3b",
      "role": "user",
      "turnIndex": 0,
      "chunkIndex": 0,
      "tokenCount": 45,
      "text": "hello"
    },
    ...
  ]
}
```

**Error cases:**
- `sessionId` not found → `{ success: false, message: "Session not found" }`
- No files match filters → `{ success: true, sessionsExtracted: 0, chunks: [] }`

### 5.2 `session_index`

Index one or more sessions: extract → chunk → embed → store.

**Input:**
```typescript
{
  sessionId?: string;      // specific session UUID
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
  force?: boolean;         // re-index even if already in DB (default false)
}
```

**Behavior:**
1. Build file list from `cwdFilter` + `dateFrom`/`dateTo` (Layer 1)
2. If `sessionId` provided, find that specific file
3. Filter out already-indexed sessions (check `session_chunks.session_id` in DB)
4. For each remaining file:
   a. Extract chunks (reuse `extract.ts` logic)
   b. Embed each chunk via `nomic-embed-text-v1.5`
   c. Store rows in `session_chunks` + vectors in `session_chunk_vectors`
5. Return stats

**Output:**
```json
{
  "success": true,
  "indexed": 3,
  "skipped": 2,
  "chunks": 47,
  "tokens": 125000,
  "errors": 0
}
```

**Error handling:**
- Embedding failure → chunk stored in DB without vector, `errors` incremented
- LMStudio unreachable → all chunks stored as text-only, `errors` = total chunks
- `force: true` → bypasses dedup check, re-indexes even if already present

### 5.3 `session_index_status`

Check what's already indexed.

**Input:** `{}`

**Output:**
```json
{
  "success": true,
  "totalSessions": 65,
  "totalChunks": 1608,
  "totalTokens": 4500000,
  "indexedSessions": [
    {
      "sessionId": "019ec5e0-...",
      "timestamp": "2026-06-14T11:24:24.260Z",
      "cwd": "/home/pi",
      "model": "qwen/qwen3.6-35b-a3b",
      "chunks": 12
    },
    ...
  ]
}
```

### 5.4 `session_search`

Semantic search across indexed conversations.

**Input:**
```typescript
{
  query: string;
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
  role?: string;           // 'user' | 'assistant'
  maxResults?: number;     // default 10
}
```

**Behavior:**
1. If `query` has content: get embedding via `nomic-embed-text-v1.5`
2. Run vector search against `session_chunk_vectors` using `sqlite-vec`
3. If vector search returns nothing or LMStudio unavailable: fallback to FTS5 text search
4. Apply `cwdFilter`, `dateFrom`, `dateTo`, `role` as WHERE clauses
5. Return top N chunks

**Output:**
```json
{
  "success": true,
  "query": "knowledge graph design",
  "method": "vector",       // or "fts5"
  "results": [
    {
      "sessionId": "019ec5e0-...",
      "sessionTimestamp": "2026-06-14T11:24:24.260Z",
      "cwd": "/home/pi",
      "model": "qwen/qwen3.6-35b-a3b",
      "role": "assistant",
      "turnIndex": 3,
      "chunkIndex": 0,
      "tokenCount": 450,
      "text": "A knowledge graph has two conflicting requirements...",
      "score": 0.87
    },
    ...
  ]
}
```

**Error cases:**
- Empty query → `{ success: false, message: "Query is required" }`
- No indexed sessions → `{ success: true, results: [] }`

### 5.5 `session_errors`

Find failed tool calls across session logs.

**Input:**
```typescript
{
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
  toolName?: string;
  maxResults?: number;     // default 20
}
```

**Behavior:**
1. Build file list from `cwdFilter` + `dateFrom`/`dateTo` (Layer 1)
2. Open matching `.jsonl` files
3. Scan for `toolResult` records with `isError: true`
4. Apply `toolName` filter
5. Return top N errors

**Output:**
```json
{
  "success": true,
  "totalCount": 199,
  "errors": [
    {
      "sessionId": "019ec5e0-...",
      "timestamp": "2026-06-14T11:24:24.260Z",
      "cwd": "/home/pi",
      "toolName": "bash",
      "toolCallId": "469528083",
      "errorMessage": "Command failed with exit code 1: ..."
    },
    ...
  ]
}
```

### 5.6 `session_stats`

Aggregate statistics across session logs.

**Input:**
```typescript
{
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
}
```

**Output:**
```json
{
  "success": true,
  "totalSessions": 65,
  "totalTurns": 222,
  "totalErrors": 199,
  "totalTokensIn": 70338067,
  "totalTokensOut": 746293,
  "toolUsage": {
    "bash": 902,
    "read": 380,
    "edit": 233,
    "kg_add": 104,
    "write": 75,
    "kg_link": 67,
    "kg_search": 54,
    "kg_get": 25,
    "kg_query": 10,
    "kg_delete": 4,
    "github_refresh_token": 1
  },
  "sessionsByProject": {
    "/home/pi": 45,
    "/home/pi/.pi/agent/extensions/kg-memory": 19,
    "/home/pi/.pi/agent/sessions": 1
  }
}
```

### 5.7 `session_list`

Browse sessions with metadata.

**Input:**
```typescript
{
  cwdFilter?: string;
  dateFrom?: string;
  dateTo?: string;
  limit?: number;          // default 50
  offset?: number;         // default 0
  sortBy?: string;         // "timestamp" | "turnCount" | "tokensIn"
}
```

**Behavior:**
1. Build file list from `cwdFilter` + `dateFrom`/`dateTo` (Layer 1: zero file reads)
2. If `sortBy` is `"turnCount"` or `"tokensIn"`: open each file to read metadata
3. Apply pagination
4. Check DB for `isIndexed` flag

**Output:**
```json
{
  "success": true,
  "total": 65,
  "sessions": [
    {
      "sessionId": "019ec5e0-...",
      "timestamp": "2026-06-14T11:24:24.260Z",
      "cwd": "/home/pi",
      "file": "2026-06-14T11-24-24-260Z_019ec5e0-...",
      "size": 45678,
      "turnCount": 16,
      "tokenCountIn": 3063938,
      "tokenCountOut": 56914,
      "errorCount": 3,
      "models": ["qwen/qwen3.6-27b", "qwen/qwen3.6-35b-a3b"],
      "isIndexed": true
    },
    ...
  ]
}
```

## 6. File Responsibilities

### `index.ts` — Entry Point

- `openDB()` → opens/initializes SQLite DB
- `registerTools(pi)` → registers all 7 tools with Pi
- No hooks, no commands, no side effects on load
- Tools are called on-demand by the agent

### `db.ts` — Database Layer

- `openDB()`: opens `~/.pi/agent/extensions/session-log-indexer/index.db`, creates tables/indexes/triggers if missing
- `chunkExists(sessionId)`: check if session is already indexed
- `insertChunks(chunks)`: bulk insert chunks and vectors
- `searchByEmbedding(embedding, maxResults, filters)`: vector search
- `searchByText(query, maxResults, filters)`: FTS5 search
- `getStatus()`: query aggregate stats
- `getIndexedSessions()`: list all indexed sessions
- `isIndexed(sessionId)`: boolean check

### `search.ts` — Embedding + Search

- `getEmbedding(text)`: call LMStudio at `http://10.1.1.145:1234/v1/embeddings` with nomic task prefix
- `batchGetEmbedding(texts)`: batch embedding call for efficiency
- `search(query, maxResults, filters)`: hybrid search (vector → FTS5 fallback)
- Same LMStudio endpoint and model as kg-memory

### `tools.ts` — Tool Implementations

Each tool is a thin function:
- Parse parameters
- Call `extract.ts`, `db.ts`, or `search.ts` as needed
- Format output as `{ content: [{ type: "text", text: JSON.stringify(result) }] }`
- Handle errors gracefully

### `extract.ts` — Session Parsing (Pure Functions)

- `parseSession(filepath)`: read `.jsonl` → records array
- `extractForeground(records)`: → (role, text) pairs, skipping non-foreground content
- `chunkTurn(role, text, chunkSize)`: → chunk array with metadata
- `getSessionMetadata(records)`: → { sessionId, timestamp, cwd, model }

**Zero side effects. Pure I/O → data transformation. Testable with mock files.**

## 7. Unit Test Plan

### `extract.test.ts`

| Test | What it verifies |
|---|---|
| `parseSession reads valid .jsonl` | Records array matches input lines |
| `parseSession skips blank lines` | No empty records |
| `parseSession handles malformed JSON` | Graceful skip, no crash |
| `extractForeground includes user text` | User text blocks are captured |
| `extractForeground includes assistant text` | Assistant text blocks are captured |
| `extractForeground excludes thinking blocks` | No thinking content in output |
| `extractForeground excludes toolCall blocks` | No toolCall content in output |
| `extractForeground excludes toolResult blocks` | No toolResult content in output |
| `extractForeground skips non-message records` | session/model_change/thinking_level_change ignored |
| `chunkTurn single chunk under 512 tokens` | One chunk, chunkIndex=0 |
| `chunkTurn multiple chunks over 512 tokens` | Sequential chunkIndex, no overlap |
| `chunkTurn exact 512 token boundary` | One chunk, no split |
| `chunkTurn very large turn (21K tokens)` | ~42 sub-chunks, correct chunkIndex |
| `chunkTurn preserves role metadata` | Role carried through to all chunks |
| `chunkTurn empty text produces no chunks` | Empty input → empty output |
| `getSessionMetadata extracts all fields` | sessionId, timestamp, cwd, model all correct |
| `getSessionMetadata handles multiple model changes` | Last model change is used |

### `search.test.ts`

| Test | What it verifies |
|---|---|
| `getEmbedding calls LMStudio with correct payload` | nomic task prefix, correct endpoint |
| `getEmbedding parses response` | Correct vector extraction |
| `getEmbedding handles LMStudio unavailable` | Returns null, no crash |
| `batchGetEmbedding sends batched request` | Single HTTP call for multiple texts |
| `search vector mode returns results` | Vector search works with mock data |
| `search FTS5 fallback works` | When vector unavailable, FTS5 returns results |
| `search applies cwdFilter` | Results scoped to directory |
| `search applies date range filter` | Results scoped to date range |
| `search applies role filter` | Only matching role returned |
| `search respects maxResults` | No more than maxResults returned |
| `search rejects empty query` | Returns error |
| `search returns empty results for empty DB` | No crash, clean empty response |

## 8. Implementation Order

1. **`package.json` + `tsconfig.json`** — scaffolding
2. **`extract.ts`** — pure functions, testable first
3. **`extract.test.ts`** — verify parsing logic
4. **`db.ts`** — SQLite setup, schema, CRUD
5. **`search.ts`** — embedding API, hybrid search
6. **`search.test.ts`** — mock LMStudio, test search logic
7. **`tools.ts`** — tool implementations
8. **`index.ts`** — wire everything together
9. **Manual integration test** — index existing sessions, run searches

## 9. Known Constraints

- **No hooks** — tools are agent-triggered only. No auto-indexing on session start.
- **No current session indexing** — the extension must not index the session it's running in. The agent controls which sessions to index via tool parameters.
- **LMStudio dependency** — embedding requires `10.1.1.145:1234`. Search falls back to FTS5-only if unavailable.
- **Filename encoding lossy** — `--home-pi-.pi-agent-extensions-kg-memory--` decodes to `/home-pi/.pi/agent/extensions/kg/memory` via simple `-` → `/` replacement, but the real path is `/home/pi/.pi/agent/extensions/kg-memory`. The session header's `cwd` field is the authoritative source.
- **Token estimation is approximate** — `charCount / 4` for chunking decisions. Actual token count stored per chunk.
