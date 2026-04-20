"""Plugin system for extending kernel code with third-party backends, profilers, skills, and hooks."""

from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import logging

logger = logging.getLogger(__name__)


class PluginType:
    BACKEND = "backend"
    PROFILER = "profiler"
    SKILL = "skill"
    HOOK = "hook"


@dataclass
class PluginSpec:
    name: str
    version: str = "0.1.0"
    plugin_type: str = PluginType.SKILL
    description: str = ""
    author: str = ""
    entry_point: str = ""  # filename in plugins dir


@dataclass
class LoadedPlugin:
    spec: PluginSpec
    module: object = None
    active: bool = True


class PluginRegistry:
    """Discovers, loads, and manages plugins from .kernel-code/plugins/."""

    def __init__(self, plugins_dir: str | Path = ".kernel-code/plugins"):
        self._dir = Path(plugins_dir)
        self._plugins: dict[str, LoadedPlugin] = {}

    def discover(self) -> list[PluginSpec]:
        """Scan plugins directory for plugin files.
        Each plugin is a .py file with a PLUGIN_SPEC dict and a register() function."""
        specs: list[PluginSpec] = []
        if not self._dir.exists():
            logger.debug("Plugins directory %s does not exist, nothing to discover.", self._dir)
            return specs

        for path in sorted(self._dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                mod_spec = importlib.util.spec_from_file_location(path.stem, path)
                if mod_spec is None or mod_spec.loader is None:
                    logger.warning("Could not create module spec for %s", path)
                    continue
                module = importlib.util.module_from_spec(mod_spec)
                mod_spec.loader.exec_module(module)

                raw = getattr(module, "PLUGIN_SPEC", None)
                if raw is None:
                    logger.warning("Plugin file %s has no PLUGIN_SPEC, skipping.", path)
                    continue

                spec = PluginSpec(
                    name=raw.get("name", path.stem),
                    version=raw.get("version", "0.1.0"),
                    plugin_type=raw.get("type", PluginType.SKILL),
                    description=raw.get("description", ""),
                    author=raw.get("author", ""),
                    entry_point=str(path),
                )
                specs.append(spec)
            except Exception:
                logger.exception("Failed to inspect plugin file %s", path)

        return specs

    def load_plugin(self, spec: PluginSpec) -> bool:
        """Load and register a plugin."""
        if spec.name in self._plugins:
            logger.info("Plugin %s is already loaded.", spec.name)
            return True

        try:
            path = Path(spec.entry_point)
            mod_spec = importlib.util.spec_from_file_location(spec.name, path)
            if mod_spec is None or mod_spec.loader is None:
                logger.error("Cannot load plugin %s: invalid module spec.", spec.name)
                return False

            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)

            register_fn = getattr(module, "register", None)
            if register_fn is None:
                logger.warning("Plugin %s has no register() function.", spec.name)
                return False

            loaded = LoadedPlugin(spec=spec, module=module, active=True)
            self._plugins[spec.name] = loaded
            logger.info("Loaded plugin: %s v%s (%s)", spec.name, spec.version, spec.plugin_type)
            return True
        except Exception:
            logger.exception("Failed to load plugin %s", spec.name)
            return False

    def unload_plugin(self, name: str):
        """Unload a plugin."""
        if name in self._plugins:
            self._plugins[name].active = False
            del self._plugins[name]
            logger.info("Unloaded plugin: %s", name)
        else:
            logger.warning("Plugin %s is not loaded, cannot unload.", name)

    def list_plugins(self) -> list[LoadedPlugin]:
        """List all discovered plugins."""
        return list(self._plugins.values())

    def get_plugin(self, name: str) -> LoadedPlugin | None:
        return self._plugins.get(name)

    def load_all(self):
        """Discover and load all plugins."""
        specs = self.discover()
        for spec in specs:
            self.load_plugin(spec)

    def call_register(self, name: str, registry_arg: object) -> bool:
        """Call the register() function of a loaded plugin, passing registry_arg."""
        loaded = self._plugins.get(name)
        if loaded is None or loaded.module is None:
            logger.warning("Plugin %s not loaded, cannot call register().", name)
            return False

        register_fn = getattr(loaded.module, "register", None)
        if register_fn is None:
            logger.warning("Plugin %s has no register() function.", name)
            return False

        try:
            register_fn(registry_arg)
            return True
        except Exception:
            logger.exception("register() failed for plugin %s", name)
            return False
