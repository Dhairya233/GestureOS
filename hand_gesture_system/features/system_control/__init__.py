"""
features/system_control/__init__.py
-------------------------------------
Exposes SystemController at the package level:

    from features.system_control import SystemController
"""

from .power_handler import SystemController

__all__ = ["SystemController"]
