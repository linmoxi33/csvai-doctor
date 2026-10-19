"""csvai - optional, removable AI assistance layer for csvkit.

The package is deliberately isolated: importing it is never required by
csvkit itself, and every public entry point degrades to a deterministic
local implementation when no model is reachable.
"""

__version__ = '0.1.0'

__all__ = ['__version__']
