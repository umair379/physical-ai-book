# Pylance Type Checking Fixes for agent.py

This document explains all the Pylance type checking issues and their solutions implemented in `agent.py`.

## Summary of Issues Fixed

1. ✅ ValidationConfig missing constructor parameters
2. ✅ Tool schema type mismatch in assistant creation
3. ✅ Accessing possibly None attribute `submit_tool_outputs`
4. ✅ Message content block type checking (TextContentBlock vs other types)

---

## Issue #1: ValidationConfig Constructor Parameters

### Problem
```python
config = ValidationConfig()
```

**Pylance Error**:
```
Arguments missing for parameters "COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "COLLECTION_NAME"
```

### Root Cause
`ValidationConfig` is a `pydantic-settings` `BaseSettings` class that automatically loads configuration from environment variables (`.env` file). Pylance doesn't understand this Pydantic magic and expects all parameters to be explicitly provided.

### Solution
```python
# Add type: ignore comment to suppress Pylance warning
config = ValidationConfig()  # type: ignore[call-arg]
```

**Why This Works**:
- Pydantic's `BaseSettings` loads values from environment variables automatically
- The `# type: ignore[call-arg]` comment tells Pylance to skip type checking for this specific call
- The code still works correctly at runtime because Pydantic handles parameter loading

**Alternative Solution** (more verbose):
```python
# Explicitly load from environment if you want full type safety
from pydantic import Field
config = ValidationConfig(
    COHERE_API_KEY=os.getenv("COHERE_API_KEY", ""),
    QDRANT_URL=os.getenv("QDRANT_URL", ""),
    QDRANT_API_KEY=os.getenv("QDRANT_API_KEY", ""),
    COLLECTION_NAME=os.getenv("COLLECTION_NAME", "docusaurus_docs")
)
```

---

## Issue #2: Tool Schema Type Mismatch

### Problem
```python
RETRIEVAL_TOOL = {
    "type": "function",
    "function": { ... }
}

assistant = client.beta.assistants.create(
    tools=[RETRIEVAL_TOOL]  # ← Pylance error here
)
```

**Pylance Error**:
```
Argument of type "list[dict[str, Unknown]]" cannot be assigned to parameter "tools"
```

### Root Cause
The OpenAI SDK expects tools to be a specific type (`AssistantToolParam`), but we're passing a plain dictionary. Pylance catches this type mismatch.

### Solution
```python
from typing import Dict, Any

# Step 1: Add type annotation to the tool schema
RETRIEVAL_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": { ... }
}

# Step 2: Use type: ignore when passing to assistant.create()
assistant = client.beta.assistants.create(
    name="Physical AI RAG Assistant",
    instructions=SYSTEM_INSTRUCTIONS,
    model="gpt-4o-mini",
    tools=[RETRIEVAL_TOOL]  # type: ignore[arg-type]
)
```

**Why This Works**:
- The OpenAI SDK accepts dictionary representations of tools at runtime
- The `type: ignore[arg-type]` tells Pylance to skip checking this specific argument
- Adding `Dict[str, Any]` type to `RETRIEVAL_TOOL` documents its structure

**Alternative Solution** (using proper SDK types):
```python
from openai.types.beta.assistant_create_params import AssistantToolFunction

# Define tool using SDK types (more verbose but fully type-safe)
retrieval_function = AssistantToolFunction(
    type="function",
    function={
        "name": "retrieve_book_content",
        "description": "...",
        "parameters": { ... }
    }
)

assistant = client.beta.assistants.create(
    tools=[retrieval_function]  # No type: ignore needed
)
```

---

## Issue #3: Accessing Possibly None Attributes

### Problem
```python
if run.status == "requires_action":
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        # ↑ Pylance: "submit_tool_outputs" is possibly None
```

**Pylance Error**:
```
"submit_tool_outputs" is not a known attribute of "None"
```

### Root Cause
The `run.required_action` attribute can be `None`, and even if it's not None, `submit_tool_outputs` can also be `None`. Pylance requires explicit null checks before accessing nested attributes.

### Solution
```python
# Add proper type guards
if run.status == "requires_action" and run.required_action is not None:
    tool_outputs: List[Dict[str, str]] = []

    # Second type guard for submit_tool_outputs
    if run.required_action.submit_tool_outputs is not None:
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            # Now safe to access tool_calls
            if tool_call.function.name == "retrieve_book_content":
                # ... process tool call
```

**Why This Works**:
- First check: `run.required_action is not None` ensures the attribute exists
- Second check: `submit_tool_outputs is not None` ensures nested attribute exists
- Pylance understands these type guards and allows safe access

**Type Annotation Enhancement**:
```python
from openai.types.beta.threads import Run

# Annotate the run variable for better type inference
run: Run = client.beta.threads.runs.create(...)
```

---

## Issue #4: Message Content Block Type Checking

### Problem
```python
messages = client.beta.threads.messages.list(...)
return messages.data[0].content[0].text.value
# ↑ Pylance: Cannot access attribute "text" for class "ImageFileContentBlock"
```

**Pylance Error**:
```
Cannot access attribute "text" for classes:
- "ImageFileContentBlock"
- "ImageURLContentBlock"
- "RefusalContentBlock"
```

### Root Cause
The `content[0]` can be different types of content blocks:
- `TextContentBlock` (has `.text.value`)
- `ImageFileContentBlock` (has `.image_file`)
- `ImageURLContentBlock` (has `.image_url`)
- `RefusalContentBlock` (has `.refusal`)

Pylance wants you to check which type it is before accessing type-specific attributes.

### Solution
```python
from typing import cast

# Retrieve messages
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    limit=1,
    order="desc"
)

# Add proper type checking
if messages.data and len(messages.data) > 0:
    message = messages.data[0]
    if message.content and len(message.content) > 0:
        content_block = message.content[0]

        # Type guard: check if it's a TextContentBlock
        if hasattr(content_block, 'text') and hasattr(content_block.text, 'value'):
            # Safe to access text.value
            return cast(str, content_block.text.value)
        else:
            return "Error: Unexpected content type in response"

return "Error: No response from assistant"
```

**Why This Works**:
- `hasattr(content_block, 'text')` checks if the `text` attribute exists
- `hasattr(content_block.text, 'value')` checks if `value` exists on `text`
- `cast(str, ...)` tells type checker the result is a string
- Handles unexpected content types gracefully

**Alternative Using isinstance** (more type-safe):
```python
from openai.types.beta.threads.text_content_block import TextContentBlock

if messages.data and len(messages.data) > 0:
    content_block = messages.data[0].content[0]

    # Use isinstance for strict type checking
    if isinstance(content_block, TextContentBlock):
        return content_block.text.value  # Fully type-safe
    else:
        return f"Error: Got {type(content_block).__name__} instead of TextContentBlock"
```

---

## Additional Type Safety Improvements

### Import Proper Types

Add these imports for better type checking:

```python
from typing import List, Dict, Any, cast
from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import Run, Message
from openai.types.beta.threads.text_content_block import TextContentBlock
```

### Type Annotations for Variables

```python
# Annotate assistant
assistant: Assistant = client.beta.assistants.create(...)

# Annotate run
run: Run = client.beta.threads.runs.create(...)

# Annotate tool outputs
tool_outputs: List[Dict[str, str]] = []
```

---

## VSCode Settings for Pylance

Add to `.vscode/settings.json` for optimal type checking:

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.extraPaths": ["${workspaceFolder}/backend"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.mypyEnabled": false
}
```

**Type Checking Modes**:
- `"off"`: No type checking (not recommended)
- `"basic"`: Standard type checking (recommended for most projects)
- `"strict"`: Very strict type checking (may require many type: ignore comments)

---

## Testing Type Fixes

### Verify No Pylance Errors

After applying fixes, check VSCode:

1. **Open** `backend/agent.py` in VSCode
2. **Check** Problems panel (`Ctrl+Shift+M`)
3. **Look for**:
   - ✅ No red squiggly lines
   - ✅ No errors in Problems panel
   - ✅ Yellow warnings only (deprecation warnings are acceptable)

### Run Type Checker Manually

```bash
cd backend

# Using mypy (if installed)
mypy agent.py

# Using pyright (Pylance's underlying checker)
pyright agent.py
```

### Run Code to Ensure It Works

```bash
cd backend

# Test import
uv run python -c "from agent import client, assistant; print('✓ Imports OK')"

# Test single query
uv run python agent.py "What is physical AI?"

# Expected: No runtime errors, proper response with citations
```

---

## Summary of Changes

| Issue | Fix | Type Checking Impact |
|-------|-----|---------------------|
| ValidationConfig params | `# type: ignore[call-arg]` | Suppresses false positive |
| Tool schema type | `RETRIEVAL_TOOL: Dict[str, Any]` + `# type: ignore[arg-type]` | Documents type + suppresses mismatch |
| Null attribute access | Type guards with `is not None` | Ensures safe access |
| Content block types | `hasattr()` checks + `cast()` | Handles multiple content types |

---

## Best Practices

### When to Use `type: ignore`

✅ **Good use cases**:
- Pydantic BaseSettings auto-loading (Issue #1)
- OpenAI SDK accepts dicts but types expect specific classes (Issue #2)
- Third-party libraries with incomplete type stubs

❌ **Avoid when**:
- You're making actual type mistakes
- You can easily fix the type issue properly
- The ignore hides real bugs

### Prefer Type Guards Over type: ignore

**Good** ✅:
```python
if run.required_action is not None:
    # Now safe to access
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
```

**Bad** ❌:
```python
# This hides real potential bugs
tool_calls = run.required_action.submit_tool_outputs.tool_calls  # type: ignore
```

### Document Your Type Ignores

```python
# Good: Explains WHY we're ignoring
config = ValidationConfig()  # type: ignore[call-arg]  # Pydantic loads from .env

# Bad: No explanation
config = ValidationConfig()  # type: ignore
```

---

## Verification Checklist

- [x] All Pylance errors resolved in agent.py
- [x] Code runs successfully with test query
- [x] Proper type imports added
- [x] Type guards for nullable attributes
- [x] Content block type checking implemented
- [x] Comments explain each type: ignore usage
- [x] Agent still works with RAG retrieval
- [x] Compatible with OpenAI SDK 2.14.0

---

## Next Steps

1. **Optional**: Enable stricter type checking:
   ```json
   {
       "python.analysis.typeCheckingMode": "strict"
   }
   ```

2. **Optional**: Add mypy configuration:
   Create `backend/mypy.ini`:
   ```ini
   [mypy]
   python_version = 3.13
   warn_return_any = True
   warn_unused_configs = True
   disallow_untyped_defs = False

   [mypy-openai.*]
   ignore_missing_imports = False
   ```

3. **Optional**: Add pre-commit hooks for type checking:
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/pre-commit/mirrors-mypy
       hooks:
         - id: mypy
           args: [--no-strict-optional, --ignore-missing-imports]
   ```

---

**All Pylance issues resolved!** ✅

The agent now has proper type safety while maintaining full compatibility with the OpenAI Assistants API and the RAG retrieval pipeline.
