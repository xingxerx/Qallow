#!/usr/bin/env python3
"""
Qallow User Listener System
Listens to user input and automatically updates the codebase based on feedback.

Location: python/listeners/user_listener.py
Purpose:
  - Monitor user interactions and feedback
  - Detect patterns and improvement opportunities
  - Trigger automatic codebase updates
  - Maintain audit trail of changes
"""

import json
import logging
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    """Types of user events that trigger updates"""
    USER_FEEDBACK = "user_feedback"
    PERFORMANCE_ISSUE = "performance_issue"
    ERROR_REPORT = "error_report"
    FEATURE_REQUEST = "feature_request"
    CONFIGURATION_CHANGE = "configuration_change"
    TELEMETRY_ANOMALY = "telemetry_anomaly"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class UserEvent:
    """Represents a user interaction event"""
    event_type: EventType
    timestamp: str
    user_id: str
    message: str
    metadata: Dict[str, Any]
    priority: int = 1  # 1-5, higher = more urgent
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "message": self.message,
            "metadata": self.metadata,
            "priority": self.priority
        }


@dataclass
class UpdateAction:
    """Represents an action to update the codebase"""
    action_id: str
    event_id: str
    action_type: str  # "config_update", "code_patch", "parameter_tune", etc.
    target_file: str
    changes: Dict[str, Any]
    timestamp: str
    status: str = "pending"  # pending, executing, completed, failed
    result: Optional[str] = None


class UserListener:
    """
    Main listener class that monitors user input and triggers updates
    """
    
    def __init__(self, data_dir: str = "data/listeners", max_queue_size: int = 1000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.update_callbacks: Dict[EventType, List[Callable]] = {}
        self.event_history: List[UserEvent] = []
        self.update_history: List[UpdateAction] = []
        
        self.logger = self._setup_logging()
        self.running = False
        self.listener_thread: Optional[threading.Thread] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the listener"""
        logger = logging.getLogger("UserListener")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.data_dir / "listener.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def register_callback(self, event_type: EventType, callback: Callable) -> None:
        """Register a callback for a specific event type"""
        if event_type not in self.update_callbacks:
            self.update_callbacks[event_type] = []
        self.update_callbacks[event_type].append(callback)
        self.logger.info(f"Registered callback for {event_type.value}")
    
    def submit_event(self, event: UserEvent) -> str:
        """Submit a user event to the listener"""
        try:
            self.event_queue.put(event, timeout=5)
            self.logger.info(f"Event submitted: {event.event_type.value}")
            return event.to_dict()
        except queue.Full:
            self.logger.error("Event queue is full, dropping event")
            return None
    
    def start(self) -> None:
        """Start the listener thread"""
        if self.running:
            self.logger.warning("Listener already running")
            return
        
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        self.logger.info("User listener started")
    
    def stop(self) -> None:
        """Stop the listener thread"""
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=5)
        self.logger.info("User listener stopped")
    
    def _listen_loop(self) -> None:
        """Main listening loop"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self._process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")
    
    def _process_event(self, event: UserEvent) -> None:
        """Process a user event and trigger callbacks"""
        self.logger.info(f"Processing event: {event.event_type.value}")
        self.event_history.append(event)
        
        # Save event to history
        self._save_event(event)
        
        # Trigger registered callbacks
        if event.event_type in self.update_callbacks:
            for callback in self.update_callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
    
    def _save_event(self, event: UserEvent) -> None:
        """Save event to disk"""
        event_file = self.data_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(event_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save event: {e}")
    
    def get_event_history(self, limit: int = 100) -> List[Dict]:
        """Get recent event history"""
        return [e.to_dict() for e in self.event_history[-limit:]]
    
    def get_update_history(self, limit: int = 100) -> List[Dict]:
        """Get recent update history"""
        return [asdict(u) for u in self.update_history[-limit:]]


# Global listener instance
_global_listener: Optional[UserListener] = None


def get_listener() -> UserListener:
    """Get or create the global listener instance"""
    global _global_listener
    if _global_listener is None:
        _global_listener = UserListener()
    return _global_listener


def submit_user_feedback(
    message: str,
    user_id: str = "system",
    event_type: EventType = EventType.USER_FEEDBACK,
    metadata: Optional[Dict] = None,
    priority: int = 1
) -> Dict:
    """Convenience function to submit user feedback"""
    listener = get_listener()
    event = UserEvent(
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        user_id=user_id,
        message=message,
        metadata=metadata or {},
        priority=priority
    )
    return listener.submit_event(event)

