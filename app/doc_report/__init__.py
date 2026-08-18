"""DOC reporting page. One view per module.

Rebuild of the PowerBI report as a single Streamlit page with a top nav.

Each view module exposes `render(ctx)` and reads nothing from module scope, so a
view can be called from the reporting page or from a test and see exactly what
it was handed.

`data.build_context` applies the page-level filters once. If two views filtered
separately they would disagree about how many deployments exist, and both
totals would be on screen at the same time.
"""
