@echo off
REM RAG Retrieval Validation Script Wrapper
REM Usage: retrieve.bat --query "your query here"

uv run python retrieve.py %*
