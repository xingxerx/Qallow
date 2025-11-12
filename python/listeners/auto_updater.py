#!/usr/bin/env python3
"""
Qallow Auto-Update Engine
Processes user feedback and automatically updates the codebase.

Location: python/listeners/auto_updater.py
Purpose:
  - Analyze user events and determine required updates
  - Apply configuration changes
  - Tune parameters based on feedback
  - Generate code patches
  - Maintain update audit trail
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from user_listener import UserEvent, EventType, UpdateAction


@dataclass
class UpdateStrategy:
    """Strategy for handling a specific type of update"""
    event_type: EventType
    pattern: str  # Regex pattern to match in event message
    action_type: str
    target_file: str
    transform_func: Optional[callable] = None


class AutoUpdater:
    """
    Automatically updates the codebase based on user feedback
    """
    
    def __init__(self, repo_root: str = ".", data_dir: str = "data/listeners"):
        self.repo_root = Path(repo_root)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.strategies: List[UpdateStrategy] = []
        self.update_history: List[UpdateAction] = []
        
        self._load_strategies()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the updater"""
        logger = logging.getLogger("AutoUpdater")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.data_dir / "updater.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_strategies(self) -> None:
        """Load update strategies"""
        self.strategies = [
            UpdateStrategy(
                event_type=EventType.CONFIGURATION_CHANGE,
                pattern=r"set\s+(\w+)\s*=\s*(.+)",
                action_type="config_update",
                target_file="qallow_config.json"
            ),
            UpdateStrategy(
                event_type=EventType.PERFORMANCE_ISSUE,
                pattern=r"slow|performance|speed|optimize",
                action_type="parameter_tune",
                target_file="config/weights.json"
            ),
            UpdateStrategy(
                event_type=EventType.ERROR_REPORT,
                pattern=r"error|fail|crash|bug",
                action_type="error_patch",
                target_file="data/listeners/error_patches.json"
            ),
        ]
        self.logger.info(f"Loaded {len(self.strategies)} update strategies")
    
    def process_event(self, event: UserEvent) -> Optional[UpdateAction]:
        """Process a user event and generate an update action"""
        self.logger.info(f"Processing event: {event.event_type.value}")
        
        for strategy in self.strategies:
            if strategy.event_type == event.event_type:
                if re.search(strategy.pattern, event.message, re.IGNORECASE):
                    return self._create_update_action(event, strategy)
        
        return None
    
    def _create_update_action(
        self,
        event: UserEvent,
        strategy: UpdateStrategy
    ) -> UpdateAction:
        """Create an update action from an event and strategy"""
        action = UpdateAction(
            action_id=f"update_{datetime.now().timestamp()}",
            event_id=event.to_dict().get("timestamp", "unknown"),
            action_type=strategy.action_type,
            target_file=strategy.target_file,
            changes=self._extract_changes(event, strategy),
            timestamp=datetime.now().isoformat(),
            status="pending"
        )
        
        self.update_history.append(action)
        self._save_update_action(action)
        
        self.logger.info(f"Created update action: {action.action_id}")
        return action
    
    def _extract_changes(
        self,
        event: UserEvent,
        strategy: UpdateStrategy
    ) -> Dict[str, Any]:
        """Extract changes from event message"""
        changes = {
            "source": "user_feedback",
            "event_type": event.event_type.value,
            "message": event.message,
            "priority": event.priority,
            "metadata": event.metadata
        }
        
        # Try to extract key-value pairs
        match = re.search(strategy.pattern, event.message, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 2:
                changes["key"] = match.group(1)
                changes["value"] = match.group(2)
        
        return changes
    
    def apply_update(self, action: UpdateAction) -> bool:
        """Apply an update action to the codebase"""
        self.logger.info(f"Applying update: {action.action_id}")
        
        try:
            target_path = self.repo_root / action.target_file
            
            if action.action_type == "config_update":
                return self._apply_config_update(target_path, action)
            elif action.action_type == "parameter_tune":
                return self._apply_parameter_tune(target_path, action)
            elif action.action_type == "error_patch":
                return self._apply_error_patch(target_path, action)
            else:
                self.logger.warning(f"Unknown action type: {action.action_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Failed to apply update: {e}")
            action.status = "failed"
            action.result = str(e)
            return False
    
    def _apply_config_update(self, target_path: Path, action: UpdateAction) -> bool:
        """Apply a configuration update"""
        try:
            if target_path.exists():
                with open(target_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update config with changes
            if "key" in action.changes and "value" in action.changes:
                config[action.changes["key"]] = action.changes["value"]
            
            with open(target_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            action.status = "completed"
            action.result = f"Updated {target_path}"
            self.logger.info(f"Config updated: {target_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Config update failed: {e}")
            return False
    
    def _apply_parameter_tune(self, target_path: Path, action: UpdateAction) -> bool:
        """Apply parameter tuning"""
        try:
            if target_path.exists():
                with open(target_path, 'r') as f:
                    weights = json.load(f)
            else:
                weights = {}
            
            # Apply tuning based on priority
            tuning_factor = 1.0 + (action.changes.get("priority", 1) * 0.05)
            
            for key in weights:
                if isinstance(weights[key], (int, float)):
                    weights[key] *= tuning_factor
            
            with open(target_path, 'w') as f:
                json.dump(weights, f, indent=2)
            
            action.status = "completed"
            action.result = f"Tuned parameters in {target_path}"
            self.logger.info(f"Parameters tuned: {target_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Parameter tuning failed: {e}")
            return False
    
    def _apply_error_patch(self, target_path: Path, action: UpdateAction) -> bool:
        """Record error patch for later analysis"""
        try:
            if target_path.exists():
                with open(target_path, 'r') as f:
                    patches = json.load(f)
            else:
                patches = []
            
            patches.append(asdict(action))
            
            with open(target_path, 'w') as f:
                json.dump(patches, f, indent=2)
            
            action.status = "completed"
            action.result = f"Error patch recorded in {target_path}"
            self.logger.info(f"Error patch recorded: {target_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error patch failed: {e}")
            return False
    
    def _save_update_action(self, action: UpdateAction) -> None:
        """Save update action to disk"""
        action_file = self.data_dir / f"updates_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(action_file, 'a') as f:
                f.write(json.dumps(asdict(action)) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save update action: {e}")
    
    def get_update_history(self, limit: int = 100) -> List[Dict]:
        """Get recent update history"""
        return [asdict(u) for u in self.update_history[-limit:]]

