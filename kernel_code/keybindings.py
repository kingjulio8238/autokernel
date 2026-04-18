"""Custom keybindings configuration for kernel code shell."""
import json
from pathlib import Path

DEFAULT_BINDINGS = {
    "ctrl-o": "/optimize --mock",
    "ctrl-s": "/show results",
    "ctrl-d": "/dashboard",
}


def load_keybindings(config_path: Path | None = None) -> dict:
    """Load keybindings from .kernel-code/keybindings.json or defaults."""
    path = config_path or Path(".kernel-code/keybindings.json")
    if path.exists():
        return json.loads(path.read_text())
    return DEFAULT_BINDINGS.copy()


def save_default_keybindings(path: Path | None = None):
    """Save default keybindings to disk."""
    path = path or Path(".kernel-code/keybindings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_BINDINGS, indent=2))
