# Journal evidence implementation plan

Design: [journal-evidence-design](../specs/2026-10-02-journal-evidence-design.md). Existing isolated worktree and push authorization continue to apply.

- [x] Root: add `task2reg/journal/verification_audit.py`, `scripts/audit_journal_verification.py`, and meaningful tests for hidden-reference execution, export-equivalent acceptance, independent patient statistics and risk bounds. Save reproducible protocol metadata. Frozen-policy mode rejects training or validation-selection overlap; exploratory val remains explicitly descriptive.
- [x] Independent implementer: add `task2reg/journal/landscape.py`, `scripts/benchmark_journal_landscape.py`, and tests for shared physical perturbations, paired results and patient-level summaries. Keep module independent of the audit.
- [x] Root: reproduce independent core-review findings and repair defects with tests. Record each change and its scientific implication.
- [x] Root: document exact commands, journal go/no-go gates, distinction between software/experimental evidence, and limits of each diagnostic. Incorporate verified primary literature from scientific review.
- [x] Independent reviewer: inspect new code for reference leakage, denominator errors, physical geometry and unsupported claims. Root resolves findings.
- [x] Root: run full tests, both diagnostic smoke runs, source-only audit and CLI checks; commit and push `research/kbs-registration-field`; verify remote SHA. Publication-level conclusions remain pending real data.

Execution: independent landscape implementation plus local audit implementation; bounded reviews proceed concurrently. No dataset acquisition or training-performance fabrication is authorized or needed.

User steering during execution: stop expanding self-audit workflows; focus on the scientific and methodological problem. The primary deliverable was revised to [separate geometric distance from task potential](../specs/2026-10-02-task-potential-design.md), with physical pose secants, positive reference-neighborhood gaps and matched M1–M6 method controls. Protocol tools already implemented remain optional support.

Verified implementation: 364 full-suite tests; CPU and CUDA AMP dual-head warm-start/training/refinement; one effective synthetic second-stage update in all six controls; actual gradient versus finite-difference alias diagnosis; oracle-UDF toy figure; standalone matched-start landscape CLI. Real patient improvements and publication readiness remain unproven.
