import config
import importlib
import sys

def load_plugins():
    sys.path.append(config.get_path('plugins-path'))
    return {p: importlib.import_module(p) for p in config.get('plugins')}
