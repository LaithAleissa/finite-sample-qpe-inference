# Presubmission plan 2: nonhardware amendment, 5 October 2026

These are follow-up sensitivity/scope analyses planned after the original outcomes; not an external preregistration. Original numerical sources, counts, thresholds and reference remain unchanged. Save this plan and numerical source hashes before new analysis. Do not tune references, seeds, selected cells or example selection after inspecting outcomes.

## QPM sensitivity

Use all eight primary cells (1,600 saved pairs) and transition eta .95/.98 at both pooled budgets (800 pairs). Base: 32M half-open energies x17 equally spaced eta; Dense:64M x33; N:32M*s x(1+16s), s=max(1,ceil(sqrt(N/256))). Weights uniform including duplicate eta=1 representatives. Dense and N coincide at N=1024 and must not be represented as independent competitors. At N=256 N equals Base. All three remain normalized outcome-independent sequence laws. Compute complete unions at the same production tolerance; retain point/set containment, lengths, topology, predictive regret and cutoff. Compare ratios of paired sample medians with 4,000 paired-bootstrap resamples, seed 2026100511, pointwise percentile intervals; do not interpret these as simultaneous or trialwise dominance.

## Synthetic circuit scope

n=4; T0; t=.8*pi/3; branch [-3.75,3.75]; u/cx transpilation level1 seed731; 512+512 shots; original three synthetic noise levels. E=1, .234375, -3.7499, adjusting only scalar offset E-1 and retaining its controlled phase. The original helper forbids a globally aliased spectrum: construct the same logical gates without that whole-spectrum guard, because only the known prepared eigenphase is modeled. Verify analytic/circuit agreement at all three energies, explicit unused-state alias scope and identical E=1 compiled circuit. Reuse E=1 records only on exact circuit/settings equality; retain all 600 originals and generate 1,200 new pairs. Seeds SeedSequence([2026100512,energy_index,level_index]). Run original CF/QPM, complete unions, exact-density distribution KL/TV fits. Adequacy on first25 pairs/cell with999 full-domain refits, plus-one p, descriptive reject<=.05; reuse the75 E=1 diagnostic records, generate150 with SeedSequence([2026100513,selected_index]). No new exact GOF/coverage claims. Example distributions: all nine exact laws, no outcome selection.

## Scientific scope

These follow-up analyses retain the original observations and reference definition. They assess sensitivity and model mismatch; they introduce no hardware evidence or exact goodness-of-fit guarantee. Their chronology is retrospective relative to the original comparisons.
