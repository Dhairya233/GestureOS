"""
features/media_control/__init__.py
------------------------------------
Exposes MediaController at the package level:

    from features.media_control import MediaController
"""

from .media_handler import MediaController

__all__ = ["MediaController"]
