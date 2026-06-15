from importlib.resources import files
from pathlib import Path

REPO_ROOT = Path(str(files("ablgpt"))).parent
