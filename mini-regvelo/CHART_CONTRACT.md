# Teaching diagnostic plots

Surface: exported PNGs, reproducible Matplotlib renderer; no interactive dashboard.
Data: this run's synthetic cells (default N=64, G=3), training epochs and 120-point
simulation grid. No external observational data, calendar dates or biological units.

| Question | Plot | Sufficiency and intended reading |
|---|---|---|
| Does optimization reduce training error? | loss line chart | >=1 epoch; one epoch gives limited evidence |
| Does time correspond to generating order? | time scatter | >=4 cells, one point per cell; no fixed claim of recovery |
| Which weights were fitted? | paired signed heatmaps | 3x3 each, shared scale, signed numeric labels |
| Do fitted states explain observations? | 2x3 abundance panels | one scatter per cell and ordered fitted line per gene |
| Are learned velocities consistent with truth? | three scatter plots | matched cell/gene samples; time-scale caveat |
| What changes after deleting a TF regulon? | 2x3 WT/KO lines | same 120 times and zero initial condition |

Palette: hard two-root cap (blue #2874A6, orange #D48628), neutral reference lines.
Use open markers vs solid lines, dashed KO/reference curves, facets and signed labels
so distinctions do not rely only on color. Heatmap is a signed diverging scale.
Footprint: 5.5--12 inches wide, 4--7 inches high, 160 DPI. File paths are results/*.png.
QA: numerical source checks, file decode checks and subsequent exported-image visual
inspection. The six charts are diagnostics, not evidence of unique parameter recovery.
