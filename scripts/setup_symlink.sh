#!/bin/bash
# Setup symlink for case-sensitivity fix

cd /home/xing/Qallow/specs/002-organize-codebase
ln -sf TASKS.md tasks.md
echo "✓ Symlink created: tasks.md -> TASKS.md"
ls -la tasks.md

