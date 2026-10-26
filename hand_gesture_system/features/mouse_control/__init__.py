"""
features/mouse_control/__init__.py
------------------------------------
Exposes MouseController at the package level:

    from features.mouse_control import MouseController
"""

from .mouse_handler import MouseController

__all__ = ["MouseController"]
