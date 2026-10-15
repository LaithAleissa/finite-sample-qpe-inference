# Bounded final scientific revision, 10 October 2026

Review-driven, locally sealed secondary/follow-up analysis; not externally preregistered.
Original scientific data and methods are retained.

1. CP: all 1600 original paired primary count datasets; alpha=.05, marginal
Clopper-Pearson failure alpha/M, equal tails. One common eta across categories.
Report complete sets, point/set inclusion, length quartiles, full/empty/disconnected
outcomes, and informational runtime. Compare CF/QPM/DIR/PLR on identical counts.
2. Enclosure: first trial of grid_noise_015, offgrid_noise_010,
halfbin_lowshots, branch_noise_010, and positive_noise_040; additionally first
transition trial at eta=.95. CF and QPM with frozen predictors and fixed Base
mixture. Use mpmath interval arithmetic at 40 decimal digits and mathematically
bounded probability envelopes; certify every discarded cell. Retain unresolved
cells. Certification applies only to these new certified outputs, not every
historical floating-point union. Compare complete certified/production unions.
3. Sample size: n=4, E=1.013 and -3.7499, eta=.10, pooled N=256,1024,4096,
16384,65536, 20 trials per cell, seed family 2026101003. Paired independent
equal-size batches. CF, Base QPM, and the existing predetermined N-adaptive
reference s=max(1,ceil(sqrt(N/256))); uniform weights, 32Ms energies and
1+16s eta values. Stream numerator evaluation in chunks. No outcome-selected
reference or sample-size extension. Report uncertainty and cost; no assumed floor.
4. Drift: n=4 and 6, E=1.013 and -3.7499, eta pairs (.10,.10),(.10,.20),
(.20,.40),(.40,.80),(.95,.98), pooled N=1024, 30 trials per cell (600 total),
seed family 2026101004. Compare shared CF and separately projected drift CF
using identical opposite-batch predictors. The exact shared set is nested in
the exact drift set. Drift handles uniform-fraction changes only. All outcomes
retained. No QPM drift extension in this bounded revision.
Numerical stopping: original bin_width/10000 production tolerance; interval
certificate tolerance bin_width/2000 with a finite budget and unresolved-cell
retention. A certificate must never silently replace failure with a success.
Existing study timings are not matched speedup measurements. Any targeted
tests and independent checks are agent-executed, not human coauthor signoff.
