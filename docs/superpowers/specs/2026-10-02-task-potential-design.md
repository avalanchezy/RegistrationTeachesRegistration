# Method revision: separate geometry from registration potential

User correction: prioritize the scientific and methodological problem; stop expanding audit workflows. The work now targets whether task-derived supervision changes optimization behavior beyond distance fitting and discrete ranking.

## Scientific decision

An exact unsigned surface distance uniquely determines its aggregated pose energy. Its derivative need not agree with same-anchor pose error, especially near repeated structures. Requiring one scalar output to remain an accurate TUDF while arbitrarily reshaping those derivatives can be contradictory. A pure gradient loss also has prior art (Gao et al., MICCAI 2020, arXiv:2003.10987); it is not our claimed invention.

Implement a controlled dual-output variant sharing the existing encoder and query trunk:

- Geometry output d=softplus(a), given physical distance reconstruction (and optional Eikonal), used for coverage and spatial pseudo targets. The shared features and differentiable geometric-coverage term still transmit task gradients; this is separation of scalar output roles, not isolation of all parameter gradients.
- Task potential u=softplus(b), used for registration fit energy, discrete ranking and local pose-energy supervision. It is not called a metric distance. Keep the physical ROI and geometry-coverage terms.
- Initialize task output from a pretrained distance output so the new registration energy initially equals the prior geometry energy. Use pseudo-Huber on both potential and geometric distance in the same energy routine; choose which is the fit term explicitly.
- Shape local energy changes with centered six-axis SE(3) probes around cached hard candidates and the manual reference. Coordinates are eta=(rho*omega,t) in mm, using the optimizer's point centroid/radius. Match central finite differences of a smooth fixed-anchor RMS error, and also match GT-neighborhood positive energy gaps to prevent stationary maxima. Finite differences use ordinary model backpropagation, avoiding unrolled second derivatives.
- Fixed default probe scale is a configurable physical length. It supervises finite-scale secants, not exact derivatives. Actual refinement remains unchanged; experiments must check its true gradients and final displacement.

This does not resolve exact source/target symmetries: any unordered pointwise scalar aggregation assigns the same energy to a transform that only permutes an identical point set. Do not claim global convexity or unique correct registration in that setting.

## Controlled experiments

From the same supervised checkpoint compare: distance+rank, distance+rank+two-step unroll, distance+rank+secant (single-head conflict control), and task-potential+rank+secant. Same architecture width, starts, supervised data, second-stage optimizer updates and inference budget. Report actual D, selection regret, basin capture, D deterioration and distance reconstruction separately. Uniform/spatial pseudo control remains secondary until supervised method gains exist.

## Implementation tasks

1. Independent math implementer: `pose_supervision.py` plus meaningful tests for physical units, origin/parity, secant targets, minima gaps and first-order parameter gradients.
2. Root: optional task head in field model, common registration query in training/inference/refinement, backward-compatible distance mode and deliberate distance-to-task initialization.
3. Root: config/engine integration, warm-start matched control configs, a trainable synthetic mechanism demonstration and concise method documentation.
4. Independent review of formula/implementation; run actual network backward, both-head train/infer smoke and existing tests; upload the authorized branch.

No real-data benefit or journal readiness is established by the synthetic demonstration.
