# Figure reproduction map

All numbered figures are supplied in PDF, SVG and PNG. Run `python -B scripts/render_figures.py` to rebuild them into a separate output tree. The numbering is extracted from the current official article and supplementary sources.

| Object | Description | Renderer | Inputs (repository-relative) |
| --- | --- | --- | --- |
| [Figure_01](main/Figure_01.pdf) | Workflow schematic | `scripts/renderers/compact.py` | Dimer â†’ QPE â†’ sets â†’ gap â†’ fixed response |
| [Figure_02](main/Figure_02.pdf) | QPE circuit | `scripts/renderers/material.py` | src/quantum_model/dimer.py |
| [Figure_03](main/Figure_03.pdf) | Five-method primary comparison | `scripts/renderers/comparators.py` | src/qpe_study/results/final_scientific/matched_comparator_summary.json; src/qpe_study/results/postreview/qpm_regret.csv |
| [Figure_04](main/Figure_04.pdf) | Identification transition | `scripts/renderers/inference.py` | src/qpe_study/results/combined_summary.csv |
| [Figure_05](main/Figure_05.pdf) | Resources and ordering | `scripts/renderers/physical.py` | src/quantum_model/results/resources.csv; src/qpe_study/results/spectral_refined/ordering_summary.csv |
| [Figure_06](main/Figure_06.pdf) | Circuit scope | `scripts/renderers/noise_reference.py` | src/qpe_study/results/presubmission2/circuit |
| [Figure_07](main/Figure_07.pdf) | Unknown-noise gap and response | `scripts/renderers/physical.py` | src/qpe_study/results/physical_bridge/summary.json; src/qpe_study/results/physical_bridge/examples.json |
| [Figure_08](main/Figure_08.pdf) | Known-channel response | `scripts/renderers/inference.py` | src/qpe_study/results/material_refined.json; src/material_model/physics.py |
| [Figure_09](main/Figure_09.pdf) | Historical fit/residuals | `scripts/renderers/material.py` | src/manuscript/data/fit_residuals.csv; src/manuscript/data/material_model/primary_fit.json |
| [Figure_10](main/Figure_10.pdf) | Uncertainty scopes | `scripts/renderers/compact.py` | src/qpe_study/results/material_refined.json; src/material_model/results/fit_sensitivity.csv; src/material_model/results/leave_one_out.csv |
| [Figure_S01](supplement/Figure_S01.pdf) | Compiled resources | `scripts/renderers/material.py` | src/manuscript/data/resources.csv |
| [Figure_S02](supplement/Figure_S02.pdf) | Known-channel comparison | `scripts/renderers/inference.py` | src/qpe_study/results/known/summary.csv |
| [Figure_S03](supplement/Figure_S03.pdf) | Candidate bootstrap | `scripts/renderers/inference.py` | src/qpe_study/results/bootstrap/summary.csv |
| [Figure_S04](supplement/Figure_S04.pdf) | Nine circuit distributions | `scripts/renderers/noise_reference.py` | src/qpe_study/results/presubmission2/circuit/distribution_diagnostics.json |
| [Figure_S05](supplement/Figure_S05.pdf) | Reference sensitivity | `scripts/renderers/noise_reference.py` | src/qpe_study/results/presubmission2/qpm/summary.csv |
| [Figure_S06](supplement/Figure_S06.pdf) | Complete seam unions | `scripts/renderers/inference.py` | src/qpe_study/results/mixture/analysis.json; src/continuous_inversion/results/sets.json |
| [Figure_S07](supplement/Figure_S07.pdf) | All response examples | `scripts/renderers/physical.py` | src/qpe_study/results/physical_bridge/examples.json |
| [Figure_S08](supplement/Figure_S08.pdf) | Equal-query comparison | `scripts/renderers/budget.py` | src/qpe_study/results/valuable_followup/budget_summary.json |
| [Figure_S09](supplement/Figure_S09.pdf) | Scaling and batch fractions | `scripts/renderers/comparators.py` | src/qpe_study/results/final_scientific/scaling_summary.json; src/qpe_study/results/final_scientific/drift_summary.json |
| [Figure_S10](supplement/Figure_S10.pdf) | Historical fit sensitivity | `scripts/renderers/material.py` | src/manuscript/data/material_model/fit_sensitivity.csv |
| [Figure_S11](supplement/Figure_S11.pdf) | Fit diagnostics | `scripts/renderers/material.py` | src/manuscript/data/material_model/gap_profile.csv; src/manuscript/data/material_model/primary_fit.json |
| [Figure_S12](supplement/Figure_S12.pdf) | TIP and anisotropy components | `scripts/renderers/material.py` | src/manuscript/data/material_model/susceptibility_ablation.csv |
