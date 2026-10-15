# Review-driven follow-up, 8 October 2026

This locally frozen follow-up is written after the original outcomes and reviewer
comments are known. It is not external preregistration or independent confirmation
of an outcome selected without that knowledge. Original counts, methods and seals
are immutable. Source and plan hashes are sealed before the follow-up runs.

1. On all 1600 original physical-bridge state datasets, compare a normalized
   all-category Dirichlet sequence predictor (Jeffreys parameters 1/2) with QPM.
   Both use pooled counts, identical unknown-uniform-noise profile likelihood,
   alpha=.025, branch, tolerance and full-union gap propagation. No multinomial
   combinatorial coefficient occurs in either sequence likelihood. Retain every
   outcome. Report inclusion, length, topology, predictive regret difference and
   measured cost; no winner is assumed. This is a paired secondary analysis.
2. Draw new independent singlet/triplet counts for five resource cells, 100 gap
   trials each. Fixed-eta cells: n=4/6 crossed with eta=.10/.95. Fifth cell: n=6,
   eta=1-(1-lambda)^63, lambda=1-.9^(1/15). Compare with the fixed n=4,.10 cell,
   which has the same per-query global depolarization strength. Shots per batch
   are 2142 (n=4) and 510 (n=6). Four batches give exactly B=128520 base controlled
   evolution queries per gap trial in every cell. This excludes state preparation,
   inverse QFT, reset/readout and compiled-gate costs. Use all four methods
   CF/QPM/DIR/PLR and the original physical map and AF restriction.
   SeedSequence([2026100805, design, state]); no outcome-based cell selection.
   Familywise Monte Carlo intervals over 20 resource and 4 paired-DIR proportions
   use .05/24. Report all trials including empty/full/disconnected outcomes.
3. Independently recompute DIR sequence numerators using 60-digit gamma functions,
   scalar nuisance truth profiles and complete gap unions. Audit selected difficult
   energy cells (branch seams, exact grid zeros, near uniformity, and boundaries
   of trial-zero output unions) against an independent high-precision finite-sum
   kernel. Check the production probability envelopes at interior sample points.
   These are targeted diagnostics, not directed-rounding certificates or a proof
   of every rejection cell or PLR global maximum. No new validity claim follows.

Uniform replacement is physically exact under the stated counting-register global
depolarization model after each base query; it does not represent arbitrary local
gate/readout noise. Explain eta_n=1-(1-lambda)^(2^n-1) and distinguish equal shots,
equal queries and equal physical noise. Keep fixed QPM and its localization-floor
limitation. Do not add hardware, imperfect-state-preparation inference or replace
the inferential target. Add results to shared article and supplement, rebuild the
existing deliverables and portable repository, and retain truthful AI disclosure.
