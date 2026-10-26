"""
features/tab_switcher/__init__.py
----------------------------------
Exposes EdgeJoystick at the package level:

    from features.tab_switcher import EdgeJoystick
"""

from .tab_handler import EdgeJoystick

# Legacy alias so any code that still imports TabSwitcher doesn't break.
TabSwitcher = EdgeJoystick

__all__ = ["EdgeJoystick", "TabSwitcher"]
