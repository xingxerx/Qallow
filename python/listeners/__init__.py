"""
Qallow User Listener System
Listens to user input and automatically updates the codebase.

This module provides:
- UserListener: Core listener for monitoring user interactions
- AutoUpdater: Engine for applying automatic updates
- QallowListenerIntegration: Integration with Qallow components
"""

from .user_listener import (
    UserListener,
    UserEvent,
    EventType,
    UpdateAction,
    get_listener,
    submit_user_feedback
)

from .auto_updater import (
    AutoUpdater,
    UpdateStrategy
)

from .qallow_listener_integration import (
    QallowListenerIntegration,
    get_integration
)

__all__ = [
    # User Listener
    "UserListener",
    "UserEvent",
    "EventType",
    "UpdateAction",
    "get_listener",
    "submit_user_feedback",
    
    # Auto Updater
    "AutoUpdater",
    "UpdateStrategy",
    
    # Integration
    "QallowListenerIntegration",
    "get_integration",
]

__version__ = "1.0.0"

