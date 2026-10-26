"""
features/modifier_key/__init__.py
-----------------------------------
Exposes ModifierController at the package level:

    from features.modifier_key import ModifierController
"""

from .modifier_handler import ModifierController

__all__ = ["ModifierController"]
