"""
Demo plots for mpl-fastapi.

This file aggregates all individual plot demos into one server.
Each plot is also available in its own standalone file:
- demos/sine.py
- demos/interactive_sine.py
- demos/cosine.py
- demos/lissajous.py
- demos/polygon.py
- demos/whoami.py
- demos/dataset_viewer.py
- demos/color_demo.py
- demos/date_range.py

Run this file with:
    python -m mpl_fastapi demos/plots.py

Or run any individual demo file:
    python -m mpl_fastapi demos/sine.py

Then visit the URL printed to the console.
"""

# Import all plot definitions from individual modules
from demos.sine import plots as sine_plots
from demos.interactive_sine import plots as interactive_sine_plots
from demos.cosine import plots as cosine_plots
from demos.lissajous import plots as lissajous_plots
from demos.polygon import plots as polygon_plots
from demos.whoami import plots as whoami_plots
from demos.dataset_viewer import plots as dataset_plots
from demos.color_demo import plots as color_demo_plots
from demos.date_range import plots as date_range_plots

# Aggregate all plots into a single registry
plots = {}
plots.update(sine_plots)
plots.update(interactive_sine_plots)
plots.update(cosine_plots)
plots.update(lissajous_plots)
plots.update(polygon_plots)
plots.update(whoami_plots)
plots.update(dataset_plots)
plots.update(color_demo_plots)
plots.update(date_range_plots)
