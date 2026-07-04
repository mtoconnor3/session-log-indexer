# Session Log Indexer

A Pi extension that indexes and semantically searches previous Pi session logs — turning raw `.jsonl` conversation files into a searchable knowledge base.

## What It Does

Session logs are the only persistent memory Pi has across sessions. This extension makes that memory **queryable** by:

- Parsing `.jsonl` session files to extract foreground conversation (excluding thinking blocks and tool calls)
- Chunking conversations at turn boundaries (~512 tokens per chunk)
- Embedding chunks via `nomic-embed-text-v1.5` through LMStudio
- Storing everything in a standalone SQLite database with FTS5 and vector search
- Providing 7 tools for the agent to index, search, and analyze session history

## Architecture

**Two-Layer Design:**

| Layer | Purpose | Tools |
|---|---|---|
| **Filesystem Index** (fast, zero-parse) | Walks `~/.pi/agent/sessions/`, parses filenames for timestamp/UUID, string-compares for date filtering | `session_list`, `session_stats`, `session_errors` |
| **Content Parser** (slower, indexed) | Opens `.jsonl` files, extracts conversation, chunks, embeds, and stores in SQLite | `session_index`, `session_extract`, `session_search`, `session_index_status` |

**Data Paths:**

- **Indexed path** (SQLite DB): `session_search`, `session_index`, `session_index_status` — user/assistant text chunks only
- **On-demand path** (raw `.jsonl` reads): `session_errors`, `session_stats`, `session_list` — tool calls, results, and metadata

## Tools

| Tool | Description |
|---|---|
| `session_extract` | Extract foreground conversation (user/assistant text only) from one or more sessions. Excludes thinking blocks, tool calls, and tool results. |
| `session_index` | Extract → chunk → embed → store. Makes session content searchable via vector and FTS5 search. Skips already-indexed sessions (idempotent). |
| `session_index_status` | Check what's already indexed: total sessions, chunks, tokens, and per-session breakdown. |
| `session_search` | Hybrid search across indexed conversations using vector embeddings (nomic-embed-text-v1.5). Falls back to FTS5 text search if embedding unavailable. |
| `session_errors` | Find failed tool calls across session logs. Scans raw `.jsonl` files for tool_result records with `isError=true`. |
| `session_stats` | Aggregate statistics across session logs: total sessions, turns, errors, tokens, tool usage, and sessions by project. |
| `session_list` | Browse sessions with metadata: turn count, token counts, error count, models used, and indexed status. |

All tools accept `cwdFilter`, `dateFrom`, and `dateTo` parameters for scoping results to a specific project or date range.

## Design Decisions

| Aspect | Choice | Rationale |
|---|---|---|
| **Chunking** | Turn boundary, ~512 tokens, `charCount / 4` estimation | Preserves conversation coherence; rough token estimation is sufficient for chunking decisions |
| **Embedding** | `nomic-embed-text-v1.5` via LMStudio (`10.1.1.145:1234`) | Lightweight, local model; no external API costs |
| **Storage** | SQLite with `better-sqlite3` + `@photostructure/sqlite-vec` | Single-file database, no external service, vector search support |
| **FTS** | FTS5 virtual table with triggers for sync | Built into SQLite, provides fast text search alongside vector search |
| **Hooks** | `session_shutdown` on quit | Auto-indexes the current session when user exits (Ctrl+D/Ctrl+C/SIGTERM). Fire-and-forget — doesn't block shutdown. Skips if already indexed. |
| **Current session** | Not indexed mid-conversation | Session file is still being written; indexing happens on clean exit via `session_shutdown` hook |
| **Oversized turns** | Split into sub-chunks (e.g., 21K token turn → ~42 sub-chunks) | Handles large assistant responses gracefully |
| **Embedding failures** | Graceful degradation to FTS5-only | Session remains searchable via text match even if embedding fails |

## Files

| File | Purpose |
|---|---|
| `extract.ts` | Pure functions: session parsing, foreground extraction, chunking, error scanning |
| `db.ts` | SQLite layer: FTS5 setup, vector search, CRUD operations, statistics |
| `search.ts` | Embedding API (LMStudio), hybrid search, batch embedding |
| `tools.ts` | 7 tool implementations wired to the Pi extension API |
| `index.ts` | Entry point — registers tools with Pi |
| `test/extract.test.ts` | 32 unit tests for extraction and chunking |
| `test/search.test.ts` | 18 unit tests for embedding and search |

## Known Limitations

- **CWD decoding:** The filename-to-path mapping is lossy; the implementation uses the session header `cwd` as authoritative
- **On-demand tools:** `session_errors`, `session_stats`, and `session_list` read raw `.jsonl` files at query time — they don't benefit from pre-computation and may be slow with thousands of sessions
- **LMStudio dependency:** Vector search requires LMStudio running locally; without it, search falls back to FTS5 text-only
