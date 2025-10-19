# Qallow Ethics Charter

Qallow exists to explore advanced autonomy under strict sustainability, compassion, and harmony constraints (E = S + C + H). Contributors must uphold the following principles:

1. **Sustainability** – Optimize for power efficiency, responsible GPU utilization, and long-term maintainability of code and documentation.
2. **Compassion** – Prioritize human oversight, humane telemetry reporting, and transparent ethics audit trails.
3. **Harmony** – Ensure modules cooperate without destabilizing emergent behaviours; respect community collaboration norms.

## Expectations

- Incorporate ethics checks into new features (`ethics_core` integration or equivalent verification).
- Maintain closed-loop telemetry (CSV + JSON) for traceability.
- Keep human feedback hooks intact; do not bypass safeguards.
- Document ethical considerations in pull requests, especially for new phases.
- Report any potential misuse to `ethics@qallow.ai`.

## Review Process

- Every PR must describe its ethics impact in the template.
- Changes to phases 8–13 require at least two maintainer approvals.
- Use `tests/integration/ethics_*` to validate PASS/FAIL thresholds.

## Violations

- Severe violations (intentional bypassing of safeguards, removal of ethics logging) may result in rejection of contributions and suspension of repository access.
- Minor infractions (missing documentation, incomplete telemetry) must be corrected before merging.
