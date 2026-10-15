# Computational validation

The release was validated on 14 October 2026 in a fresh Python 3.12 environment installed from `requirements.txt`:

- All 85 scientific unit tests passed.
- All 27 numbered figures (11 main and 16 supplementary) regenerated from saved observations and results in PDF, SVG and PNG.
- The independent numerical audit passed: 1,600 CP decisions, 600 batch-drift decisions, 400 scaling-QPM decisions and 800 fresh shared-CF decisions.
- Stored partition records for twelve selected directed-interval certificates were checked. The historical production collection is not globally certified.
- All documented figure input paths and relative Markdown links resolve.
- Distributed files are covered by the SHA-256 manifest.

The expensive production inference collection was not rerun for this release. Numerical tests and audits do not constitute independent human/coauthor verification. Dated protocols and historical execution seals retain the chronology and scope of the studies; `source_relocation.json` records the original relocation, not the current release manifest.
