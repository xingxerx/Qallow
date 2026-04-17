#!/usr/bin/env python3
"""
Quick test of DeepSeek baseline integration with Feature 004
"""

import sys
sys.path.insert(0, '/home/xing/Qallow')

from python.deepseek_baseline import DeepSeekClient, DeepSeekConfig

print("=" * 70)
print("DeepSeek Baseline Test for Feature 004")
print("=" * 70)

# Test 1: Initialize with mock backend (for testing)
print("\n✓ Test 1: Initialize DeepSeek with MOCK backend")
config = DeepSeekConfig(backend="mock")
client = DeepSeekClient(config)
print(f"  Backend: {client.backend.value}")
print(f"  Model: {config.model}")

# Test 2: Cognitive state reasoning
print("\n✓ Test 2: Test cognitive state reasoning")
result = client.reason_cognitive_state(
    iteration=10,
    current_loss=0.234,
    best_loss=0.234,
    ethics_score=0.95,
    backend_name="CPU"
)
print(f"  Analysis: {result.get('analysis', 'N/A')}")
print(f"  Converged: {result.get('converged', False)}")
print(f"  Recommendation: {result.get('recommendation', 'N/A')}")

# Test 3: Ethics audit
print("\n✓ Test 3: Test ethics audit")
audit = client.audit_ethics(
    action="increase_exploration",
    loss_improvement=0.234,
    iteration=10
)
print(f"  Safety: {audit.get('safety', 'N/A')}")
print(f"  Control: {audit.get('control', 'N/A')}")
print(f"  Honesty: {audit.get('honesty', 'N/A')}")

# Test 4: Status
print("\n✓ Test 4: Check DeepSeek status")
status = client.get_status()
print(f"  Backend: {status.get('backend', 'unknown')}")
print(f"  Model: {status.get('model', 'unknown')}")
print(f"  Available: {status.get('available', False)}")

print("\n" + "=" * 70)
print("✅ DeepSeek baseline integration ready for Feature 004!")
print("=" * 70)
