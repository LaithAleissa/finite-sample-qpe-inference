# qpe_study final nonhardware analysis plan, 8 October 2026

Frozen before generating the new counts or examining their outcomes. This local
hash seal is not external preregistration. Earlier studies are inherited inputs.

The new study links unknown-uniform-replacement energy inference to the copper
acetate case. Four cells are fixed: (n,eta)=(4,.10),(6,.10),(4,.95),(6,.95).
Each has 200 independent trials with S at E=-3 and T0 at E=1, J=1, h=c=0,
t=0.8377580409572781, and branch [-3.75,3.75]. Each state has two independent
512-shot multinomial batches; 2048 total shots per gap trial. Noise is shared
within a state's batches and is unknown to every fitted method. Separate state
fits do not assume or exploit equality of the two states' nuisance parameters.

CF, original fixed QPM (32M half-open energies by 17 eta values, uniform
weights), and conventional PLR use alpha=.025 per state. The full difference
union has at least .95 model-conditional coverage for CF and QPM by a union
bound. No such finite-sample claim is made for PLR. The existing common Taylor/
global envelope, bin-width/10000 terminal width and 300000-cell limit apply.
Retain raw gap unions and their intersection with the declared AF prior [0,7.5].
This physically declared positive-coupling restriction is used only downstream;
it is not used to train fits or select energy components. Zero is included as
the AF limiting response. No wrapping of difference intervals is performed.

The classical per-Cu powder spin response holds g_parallel=2.42,
g_perpendicular=2.08 and scale 315*(hc/kB)/4 K/model unit fixed. Exclude TIP,
impurity and anisotropy from this sampling-only propagation. Monotonicity in the
gap maps every accepted interval to a response interval. Save full gap unions,
response images at T=50,100,200,283,300,500 K for every trial, and dense curves
on T=50,...,500 K for trial zero in each cell. Empty/unresolved outcomes remain.
Report gap inclusion, response inclusion, raw and AF-restricted total lengths,
IQRs, component counts, full-AF/empty frequencies, and relative response-hull
width at 283 K. Monte Carlo binomial intervals are simultaneous across the
12 method-by-cell inclusion proportions, not across inherited study families.
Curve inclusion over all temperatures follows from the fixed response map;
finite-grid numerical replay checks implementation but does not prove that fact.

Independent raw-count seed: SeedSequence([2026100804,design_index,state_index]).
All methods see identical counts; no seed, cell or example tuning is allowed.
Source hashes and the plan hash precede raw-count generation; raw hashes precede
inference. Save failures and stop instead of delivering partial numerical sets.
New results may contradict earlier expectations and must still be reported.

Restore early physical motivation and a dependency-aware workflow, the compact
logical resource comparison and refined known-channel ordering plot. Preserve
the four substantive physical figures. Integrate the new study as the main
gap-to-response link; label the old known-channel example as a separate check.
Consolidate repeated diagnostics without strengthening efficiency, hardware,
material-discovery or numerical-certification claims. No hardware experiments are included.
