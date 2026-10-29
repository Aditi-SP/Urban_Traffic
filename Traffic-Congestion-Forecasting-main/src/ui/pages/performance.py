import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # project root, so the page also works when run on its own
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ui import performance_sections

performance_sections.render()
