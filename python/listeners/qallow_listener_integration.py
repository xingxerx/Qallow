#!/usr/bin/env python3
"""
Qallow Listener Integration
Integrates the user listener system with Qallow components.

Location: python/listeners/qallow_listener_integration.py
Purpose:
  - Connect listener to Qallow agents
  - Integrate with telemetry system
  - Hook into memory system
  - Provide API for external feedback
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from user_listener import (
    UserListener, UserEvent, EventType, get_listener, submit_user_feedback
)
from auto_updater import AutoUpdater


class QallowListenerIntegration:
    """
    Integrates the listener system with Qallow components
    """
    
    def __init__(self, repo_root: str = ".", data_dir: str = "data/listeners"):
        self.repo_root = Path(repo_root)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.listener = get_listener()
        self.updater = AutoUpdater(repo_root, data_dir)
        
        self.logger = self._setup_logging()
        self._setup_callbacks()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("QallowListenerIntegration")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.data_dir / "integration.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_callbacks(self) -> None:
        """Setup callbacks for different event types"""
        self.listener.register_callback(
            EventType.USER_FEEDBACK,
            self._handle_user_feedback
        )
        self.listener.register_callback(
            EventType.PERFORMANCE_ISSUE,
            self._handle_performance_issue
        )
        self.listener.register_callback(
            EventType.ERROR_REPORT,
            self._handle_error_report
        )
        self.listener.register_callback(
            EventType.CONFIGURATION_CHANGE,
            self._handle_configuration_change
        )
        self.listener.register_callback(
            EventType.TELEMETRY_ANOMALY,
            self._handle_telemetry_anomaly
        )
        
        self.logger.info("Callbacks registered")
    
    def _handle_user_feedback(self, event: UserEvent) -> None:
        """Handle user feedback events"""
        self.logger.info(f"Handling user feedback: {event.message}")
        
        # Process with auto-updater
        action = self.updater.process_event(event)
        if action:
            self.logger.info(f"Generated update action: {action.action_id}")
            # Could apply automatically or queue for review
            if event.priority >= 4:
                self.updater.apply_update(action)
    
    def _handle_performance_issue(self, event: UserEvent) -> None:
        """Handle performance issue reports"""
        self.logger.info(f"Handling performance issue: {event.message}")
        
        # Extract performance metrics from metadata
        metrics = event.metadata.get("metrics", {})
        
        # Log to telemetry
        self._log_to_telemetry("performance_issue", metrics)
        
        # Process with auto-updater
        action = self.updater.process_event(event)
        if action:
            self.updater.apply_update(action)
    
    def _handle_error_report(self, event: UserEvent) -> None:
        """Handle error reports"""
        self.logger.error(f"Error reported: {event.message}")
        
        # Extract error details
        error_details = event.metadata.get("error", {})
        
        # Log to error tracking
        self._log_error(event.message, error_details)
        
        # Process with auto-updater
        action = self.updater.process_event(event)
        if action:
            self.updater.apply_update(action)
    
    def _handle_configuration_change(self, event: UserEvent) -> None:
        """Handle configuration change requests"""
        self.logger.info(f"Handling config change: {event.message}")
        
        # Process with auto-updater
        action = self.updater.process_event(event)
        if action:
            self.updater.apply_update(action)
    
    def _handle_telemetry_anomaly(self, event: UserEvent) -> None:
        """Handle telemetry anomalies"""
        self.logger.warning(f"Telemetry anomaly detected: {event.message}")
        
        # Extract anomaly data
        anomaly_data = event.metadata.get("anomaly", {})
        
        # Log anomaly
        self._log_anomaly(anomaly_data)
        
        # Process with auto-updater
        action = self.updater.process_event(event)
        if action:
            self.updater.apply_update(action)
    
    def _log_to_telemetry(self, event_type: str, data: Dict) -> None:
        """Log event to telemetry system"""
        try:
            telemetry_file = self.data_dir / "telemetry_events.jsonl"
            with open(telemetry_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "data": data
                }
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log telemetry: {e}")
    
    def _log_error(self, message: str, details: Dict) -> None:
        """Log error to error tracking"""
        try:
            error_file = self.data_dir / "error_log.jsonl"
            with open(error_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "details": details
                }
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")
    
    def _log_anomaly(self, anomaly_data: Dict) -> None:
        """Log anomaly detection"""
        try:
            anomaly_file = self.data_dir / "anomalies.jsonl"
            with open(anomaly_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "anomaly": anomaly_data
                }
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log anomaly: {e}")
    
    def start(self) -> None:
        """Start the listener system"""
        self.listener.start()
        self.logger.info("Qallow listener integration started")
    
    def stop(self) -> None:
        """Stop the listener system"""
        self.listener.stop()
        self.logger.info("Qallow listener integration stopped")
    
    def submit_feedback(
        self,
        message: str,
        event_type: EventType = EventType.USER_FEEDBACK,
        metadata: Optional[Dict] = None,
        priority: int = 1,
        user_id: str = "system"
    ) -> Dict:
        """Submit feedback to the listener"""
        return submit_user_feedback(
            message=message,
            user_id=user_id,
            event_type=event_type,
            metadata=metadata,
            priority=priority
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the listener system"""
        return {
            "listener_running": self.listener.running,
            "event_history_size": len(self.listener.event_history),
            "update_history_size": len(self.updater.update_history),
            "strategies_loaded": len(self.updater.strategies),
            "timestamp": datetime.now().isoformat()
        }


# Global integration instance
_global_integration: Optional[QallowListenerIntegration] = None


def get_integration(repo_root: str = ".") -> QallowListenerIntegration:
    """Get or create the global integration instance"""
    global _global_integration
    if _global_integration is None:
        _global_integration = QallowListenerIntegration(repo_root)
    return _global_integration

