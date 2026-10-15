# Figure renderers

Called by `../render_figures.py`; pass an output asset directory. Inference renders the known-channel/transition/alias/bootstrap and propagation plots. Physical renders resources/ordering and unknown-noise gap/response plots. Noise-reference renders grid sensitivity and circuit scope. Comparators renders the final five-method comparison and sample-size/drift figure. Budget renders equal-query results. Material renders the circuit and historical-response plots. Compact renders the IEEE column-sized workflow and uncertainty-scope figures.

Clarity renders the final comparison and response illustrations with explicit method, ensemble and sensitivity labels.

These scripts contain plotting code only and write exclusively to their requested output asset directory.
