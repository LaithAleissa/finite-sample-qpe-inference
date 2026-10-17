# QPE inference with unknown uniform replacement

Computational companion to **Finite-Sample QPE Inference With Unknown Uniform Replacement: A Copper Acetate Dimer Case Study**.

This repository contains the scientific code, integer observations, complete numerical confidence sets, derived results, and all figures used in the article and its supplementary material. Start with the numbered figures or rebuild them from the saved evidence. The manuscript, review correspondence, earlier drafts, publisher templates, and development archives are kept outside this repository.

## Layout

| Folder | Contents |
| --- | --- |
| [`figures/main`](figures/main) | Figures 1–11 in PDF, SVG and PNG |
| [`figures/supplement`](figures/supplement) | Numbered supplementary figures in PDF, SVG and PNG |
| [`src`](src) | Scientific implementations and authoritative observations/results |
| [`scripts`](scripts) | Figure regeneration, integrity checks, tests, and study replay |

## Setup and quick checks

Use Python 3.12. From the repository root:

```text
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`; macOS/Linux: `source .venv/bin/activate`), then run:

```text
python -m pip install -r requirements.txt
python -B scripts/check.py hashes
python -B scripts/check.py tests
python -B scripts/render_figures.py
```

The figure command writes to `reproductions/figures`; supplied figures and observations are preserved. An installed scientific environment can also run these commands directly. See [reproduction instructions](scripts/README.md), the [figure map](figures/README.md), and the [evidence index](src/qpe_study/results/README.md).

## Validation

All 85 scientific tests passed, every numbered figure was regenerated, and the independent final-study audit passed in this reduced tree. See [validation details](VALIDATION.md).

## Scientific scope

CF, QPM, DIR and simultaneous CP compare confidence constructions under a specified QPE count law. PLR has approximate calibration. Batch-specific CF allows unequal uniform replacement fractions; it does not cover arbitrary circuit errors. The circuit-noise experiments are model-misspecification diagnostics. Copper-acetate susceptibility is a classical response conditional on external calibration and model assumptions. Twelve selected numerical outer sets have directed-interval certificates; historical sets are not globally certified. The computational checks do not constitute independent human verification.

## License and data provenance

The software license and its scope are provided in [LICENSE](LICENSE). Historical digitized susceptibility inputs retain their source attribution and documented provenance limitations. [CITATION.cff](CITATION.cff) provides repository citation metadata.

Existing software copyright notices are retained. Software copyright attribution is distinct from the associated article's author list, which is Laith Al-Issa and Abdalkarim Awad.

## Manuscript figure correspondence

The article source references `figs/Figure_01.pdf` through `figs/Figure_11.pdf`; these are byte-identical to the PDFs in `figures/main/`. The supplement similarly references `figs/Figure_S01.pdf` through `figs/Figure_S16.pdf`, matching `figures/supplement/`. Publisher assets and author photographs belong to the separate manuscript source package, not this computational repository.
