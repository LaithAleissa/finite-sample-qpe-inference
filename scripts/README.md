# Reproduction commands

Run from the repository root with the pinned Python environment.

| Command | Purpose |
| --- | --- |
| `python -B scripts/check.py hashes` | Verify every distributed file against the manifest |
| `python -B scripts/check.py tests` | Run the scientific unit-test suites |
| `python -B scripts/check.py audit` | Independently replay CP, drift and scaling truth decisions and inspect selected certificates; several minutes |
| `python -B scripts/render_figures.py --output reproductions/figures` | Rebuild all numbered main/supplement figures from saved evidence |
| `python -B scripts/run_study.py cp --output reproductions/cp --workers 1` | Recompute the 1,600 CP sets in a separate tree |
| `python -B scripts/run_study.py scaling --output reproductions/scaling --workers 1` | Regenerate the 200 scaling datasets and recompute inference |
| `python -B scripts/run_study.py drift --output reproductions/drift --workers 1` | Regenerate the 600 batch-fraction datasets and recompute inference |

Choose a new output directory for every rendering/replay. Full study reruns can be expensive; growing-grid mixtures are particularly costly at large sample sizes. Worker count changes scheduling, not the frozen seeds. Study replay copies the scientific evidence tree and avoids the original manuscript snapshot routine; it preserves scientific source bytes and the supplied evidence. No checkpoints are distributed, so a rerun recomputes all requested sets.

The figure renderers are plotting-only extracts of the original analysis programs. They retain the saved-data calculations and plotting conventions while omitting manuscript-writing operations. PDF metadata, SVG identifiers and minor rendering differences can depend on the platform; numerical input agreement matters more than identical regenerated image bytes.

Other study implementations remain under `src/qpe_study/code` for inspection and controlled use. Their direct runners may write results in place; work on a copy if rerunning them. Root commands provide the recommended safe route. No TeX installation is needed.
