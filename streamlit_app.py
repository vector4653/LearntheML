"""
Streamlit Cloud Entry Point (Repository Root)
==============================================
Delegates execution cleanly to app/streamlit_app.py.
Ensures repository root and app package are properly resolved on Streamlit Cloud.
"""

from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parent
_APP_DIR = _ROOT / "app"

for _p in [str(_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

target_script = _APP_DIR / "streamlit_app.py"
runpy.run_path(str(target_script), run_name="__main__")
