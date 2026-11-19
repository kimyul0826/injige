"""Workspace-only dependency bootstrap (venv rejects ':' in this workspace path)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent / '.deps'))
from llm_astar.__main__ import main
if __name__ == '__main__':
    main()
