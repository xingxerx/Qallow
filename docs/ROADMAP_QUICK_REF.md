# Roadmap Quick Reference

## 🎯 Top 3 Priorities (Next Actions)

### 1. 🔴 Variable Naming Cleanup (6-10 hrs)
```bash
# Refactor these poor names across codebase:
# t → tick_index / time_step
# c → char_ptr / current_char  
# h → head_index / height
# w → width
# v → parsed_value
# p → pattern_idx
# e → error_value
# g → global_stability

cd /home/xing/Qallow
grep -r "= [a-z] /*" --include="*.c" | head -20  # See all
```

### 2. 🔴 CUDA Parallel Execution (8-16 hrs)
```bash
# Create backend/cuda/multi_pocket.cu with:
# - GPU kernel for parallel pocket state propagation
# - CUDA stream management
# - Memory coalescing optimization

# Test with: QALLOW_ENABLE_CUDA=ON cmake -B build && ctest
```

### 3. 🔴 Semantic Memory Persistence (4-8 hrs)
```bash
# Upgrade backend/cpu/semantic_memory.c from memory-only to LMDB
# - Install LMDB: apt-get install liblmdb-dev
# - Integrate LMDB snapshots
# - Test crash recovery
```

---

## 📊 Timeline Estimate

| Phase | Items | Hours | Target Date |
|-------|-------|-------|------------|
| Week 1 | Critical (3 items) | 20-34 | Nov 11 |
| Week 2 | High (3 items) | 14-22 | Nov 18 |
| Week 3 | Medium (4 items) | 22-34 | Nov 25 |
| Week 4 | Low (2 items) | 8-11 | Dec 2 |

**Total: 64-101 hours over 4 weeks**

---

## ✅ Quick Wins (< 2 Hours Each)

1. Rename `t` → `tick_index` in 2 files
2. Rename `c` → `char_ptr` in 3 files
3. Add malloc-loop comments in backend files
4. Update error handling in launcher.c

---

## 📚 Related Documentation

- Full roadmap: `docs/TODO_ROADMAP.md`
- Architecture: `docs/ARCHITECTURE_SPEC.md`
- Build guide: `README.md`
- API docs: `docs/QUICKSTART.md`

---

## 🤝 How to Contribute

```bash
# 1. Pick an item from roadmap
# 2. Create feature branch
git checkout -b feature/roadmap-X-description

# 3. Make changes
# 4. Run tests
ctest --test-dir build -V

# 5. Commit with reference
git commit -m "feat(roadmap-X): Description of change"

# 6. Push and create PR
git push origin feature/roadmap-X-description
```

---

## 📞 Contact

- **Questions?** Check TODO_ROADMAP.md for full context
- **Want to own an item?** Update the "Owners" field in roadmap
- **Found a blocker?** Document it as a new roadmap item

**Last updated: November 4, 2025**
