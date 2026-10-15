# Observations and results

| Folder/file | Evidence |
| --- | --- |
| mixture; combined_summary.csv | Original unknown-noise QPM outcomes and paired comparisons |
| known | Common-envelope known-channel comparison |
| replication | Fresh targeted replications |
| bootstrap; adequacy | Candidate bootstrap and saved adequacy resample statistics |
| circuit | Initial exact circuit-noise distributions and outcomes |
| presubmission2/qpm | Base, Dense and N-adaptive reference sensitivity |
| presubmission2/circuit; presubmission2/adequacy | Nine phase/noise cells and all 225 diagnostic records |
| postreview | Predictive regret and paired bootstrap summaries |
| spectral_refined | Complete known-channel gap/ordering outcomes |
| material_refined.json | Refined known-channel gap and fixed material response inputs |
| physical_bridge | 800 unknown-noise two-state trials, raw counts, full unions and response images |
| primary_dirichlet | Matched DIR on all 1,600 original primary datasets |
| valuable_followup | Physical DIR and 500 equal-query gap trials |
| final_scientific | CP on 1,600 datasets; 200 scaling and 600 drift datasets; all outcomes and 12 certificates |

Raw integer counts are separate from complete inference records and derived summaries. Generating energies are used for validation/scoring, not to restrict inference. Times are recorded execution costs, not matched hardware benchmarks. Per-trial temporary checkpoints are omitted; complete consolidated records remain. `seal.json` records frozen inputs/source hashes, while `manifest.json` at the root verifies the distributed package.
