"""
features/app_launcher/__init__.py
-----------------------------------
Exposes AppLauncher at the package level:

    from features.app_launcher import AppLauncher
"""

from .launcher_handler import AppLauncher

__all__ = ["AppLauncher"]
