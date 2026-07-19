# RAG Retrieval Validation Script

## Quick Start

### Running the Script

**IMPORTANT**: Always use `uv run` to execute the script. This ensures all dependencies are available.

```bash
# Navigate to backend directory
cd backend

# Run with a query
uv run python retrieve.py --query "What is physical AI?"

# Run with custom top-k results
uv run python retrieve.py --query "ROS 2 basics" --top-k 5

# Run with verbose logging
uv run python retrieve.py --query "computer vision" --verbose
```

### Why use `uv run`?

The script requires dependencies (pydantic-settings, cohere, qdrant-client, etc.) that are managed by `uv`. Running with plain `python` will fail with:

```
ModuleNotFoundError: No module named 'pydantic_settings'
```

**Solution**: Always prefix with `uv run`

### Alternative: Batch File (Windows)

Use the provided batch wrapper for convenience:

```bash
retrieve.bat --query "What is physical AI?"
```

## Usage Examples

### Basic Query
```bash
uv run python retrieve.py --query "What is physical AI?"
```

**Output**:
- Connects to Qdrant (verifies 192 vectors)
- Generates query embedding via Cohere
- Returns top-3 most similar chunks
- Displays scores, titles, URLs, text previews

### Module-Specific Query
```bash
uv run python retrieve.py --query "ROS 2 basics" --top-k 5
```

**Output**: Returns 5 results focused on ROS 2 fundamentals and Module 1

### Verbose Mode
```bash
uv run python retrieve.py --query "neural networks" --verbose
```

**Output**: Includes debug logs (embedding dimensions, API calls, detailed timing)

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--query` | Single test query to execute | Required (or --test-suite) |
| `--test-suite` | JSON file with test queries | Optional |
| `--top-k` | Number of results to return | 3 |
| `--verbose` | Enable debug logging | False |

## Environment Variables

The script requires a `.env` file in the `backend/` directory with:

```bash
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=docusaurus_docs
```

These are automatically loaded from Feature 008.

## Output

### Console Output
- Connection status and collection metadata
- Query execution progress
- Top-k results with scores, titles, URLs, text previews
- Validation status

### Log File
- Timestamped log file: `validation_YYYYMMDD_HHMMSS.log`
- Contains detailed execution logs (API calls, timings, errors)
- Located in `backend/` directory

## Success Criteria

✅ **SC-001**: Connection successful, 192 vectors verified
✅ **SC-002**: Results manually inspected, relevant content confirmed
✅ **SC-003**: Queries return results >0.4 similarity

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pydantic_settings'`

**Cause**: Running with plain `python` instead of `uv run`

**Fix**: Use `uv run python retrieve.py ...`

### Error: `Configuration validation failed`

**Cause**: Missing or invalid `.env` file

**Fix**: Ensure `.env` exists in `backend/` with required variables

### Error: `Connection failed`

**Cause**: Invalid Qdrant credentials or network issues

**Fix**: Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

## Implementation Details

- **Script**: `backend/retrieve.py` (551 lines)
- **Dependencies**: pydantic-settings, cohere, qdrant-client, python-dotenv
- **Architecture**: Single-file functional design
- **Embedding**: Cohere embed-english-v3.0 with `input_type='search_query'`
- **Search**: Qdrant `query_points()` API (timeout=120s)
- **Logging**: Dual-format (console + file)

## Next Steps

- Run validation scenarios from `specs/009-rag-retrieval/quickstart.md`
- Test with various query types (module-specific, edge cases, adversarial)
- Implement US3 (metadata validation) and US4 (performance tracking)
