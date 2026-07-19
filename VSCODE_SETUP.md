# VSCode Python Environment Setup Guide

## Issue
VSCode shows: `Import "openai" could not be resolved` for `from openai import OpenAI`

## Root Cause
VSCode is using a different Python interpreter than the virtual environment where packages are installed.

## ✅ Automated Fix Applied

The following has been configured automatically:
- Created `.vscode/settings.json` with correct Python interpreter path
- Set interpreter to: `backend/.venv/Scripts/python.exe`
- Configured Python analysis paths

## 📋 Manual Steps Required in VSCode

### Step 1: Select Python Interpreter

**Option A - Command Palette (Recommended):**
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: `Python: Select Interpreter`
3. Choose: `Python 3.13.x ('backend/.venv': venv)` or the one showing path `.\backend\.venv\Scripts\python.exe`

**Option B - Status Bar:**
1. Look at bottom-left of VSCode window
2. Click on the Python version shown (e.g., "Python 3.13.x")
3. Select: `.\backend\.venv\Scripts\python.exe`

### Step 2: Reload VSCode Window

After selecting the interpreter:
1. Press `Ctrl+Shift+P` → `Developer: Reload Window`
2. Or close and reopen VSCode

### Step 3: Verify Import Resolution

Open `backend/agent.py` and check:
- ✅ No red squiggly line under `from openai import OpenAI`
- ✅ IntelliSense/autocomplete works for OpenAI classes
- ✅ Hover over `OpenAI` shows type information

## 🔧 Alternative Installation Methods

### Method 1: Using UV (Current Setup) ✅
```bash
cd backend
uv add openai
```
**Status**: Already completed - `openai@2.14.0` installed

### Method 2: Using pip with venv
```bash
cd backend
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install openai
```

### Method 3: Using pip directly (if venv activation fails)
```bash
cd backend
.venv\Scripts\python.exe -m pip install openai  # Windows
# .venv/bin/python -m pip install openai  # Linux/Mac
```

## 📦 Installed Packages Verification

Run this command to verify all packages:
```bash
cd backend
uv run python -c "import openai, cohere, qdrant_client; print('All imports OK')"
```

**Expected output**: `All imports OK`

**Current installation status**:
```
openai             2.14.0  ✓
cohere             5.20.1  ✓
qdrant-client      1.16.2  ✓
pydantic           2.12.5  ✓
pydantic-settings  2.12.0  ✓
python-dotenv      1.2.1   ✓
```

## 🔍 Troubleshooting

### Issue: Still showing import error after selecting interpreter

**Solution 1 - Restart Python Language Server:**
1. `Ctrl+Shift+P` → `Python: Restart Language Server`

**Solution 2 - Clear VSCode cache:**
1. Close VSCode
2. Delete: `%APPDATA%\Code\Cache` (Windows) or `~/.config/Code/Cache` (Linux)
3. Reopen VSCode

**Solution 3 - Reinstall Python extension:**
1. Go to Extensions (`Ctrl+Shift+X`)
2. Search: "Python"
3. Click "Reload" or "Reinstall"

### Issue: Multiple Python interpreters showing

**Solution:**
Look for the one with path ending in `\backend\.venv\Scripts\python.exe`

### Issue: Virtual environment not showing in list

**Solution - Create interpreter setting manually:**
1. Create/edit `.vscode/settings.json` (already done)
2. Verify path: `"python.defaultInterpreterPath": "${workspaceFolder}/backend/.venv/Scripts/python.exe"`
3. Reload VSCode

## 🧪 Test Your Setup

### Test 1: Import Test
```bash
cd backend
uv run python -c "from openai import OpenAI; print('Import successful')"
```

### Test 2: Run Agent
```bash
cd backend
uv run python agent.py "What is physical AI?"
```

### Test 3: VSCode Terminal
Open VSCode integrated terminal and run:
```bash
cd backend
python -c "import openai; print(openai.__version__)"
```

Should output: `2.14.0`

## 📁 Project Structure

```
physical-ai-book/
├── .vscode/
│   └── settings.json          # ← VSCode configuration (created)
├── backend/
│   ├── .venv/                 # ← Virtual environment (use this interpreter!)
│   │   ├── Scripts/
│   │   │   └── python.exe     # ← Correct Python interpreter
│   │   └── Lib/
│   │       └── site-packages/
│   │           └── openai/    # ← Package installed here
│   ├── agent.py               # ← Your code
│   ├── retrieve.py
│   ├── .env
│   └── pyproject.toml
└── specs/
```

## ✅ Success Checklist

- [x] `.vscode/settings.json` created with correct interpreter path
- [ ] Selected interpreter in VSCode (`Ctrl+Shift+P` → `Python: Select Interpreter`)
- [ ] Reloaded VSCode window
- [ ] No import errors in `agent.py`
- [ ] IntelliSense works for `openai` module
- [ ] Can run `uv run python agent.py` successfully

## 📚 Additional Resources

- **VSCode Python Setup**: https://code.visualstudio.com/docs/python/environments
- **Virtual Environments**: https://docs.python.org/3/tutorial/venv.html
- **UV Package Manager**: https://github.com/astral-sh/uv

## 🆘 Still Having Issues?

If you've followed all steps and still see import errors:

1. **Check Python version compatibility**:
   ```bash
   cd backend
   python --version
   ```
   Should be Python 3.13 or compatible

2. **Verify package actually exists**:
   ```bash
   cd backend
   .venv\Scripts\python.exe -m pip show openai
   ```

3. **Check VSCode Python extension version**:
   - Should be v2023.x or newer
   - Update if outdated

4. **Last resort - Recreate virtual environment**:
   ```bash
   cd backend
   rm -rf .venv
   uv sync
   ```
   Then reselect interpreter in VSCode

---

**Quick Fix Summary**:
1. Open Command Palette (`Ctrl+Shift+P`)
2. Type: "Python: Select Interpreter"
3. Choose: `.\backend\.venv\Scripts\python.exe`
4. Reload window
5. ✅ Done!
