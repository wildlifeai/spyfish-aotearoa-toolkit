"""Chart implementations, shared by the views that draw them.

Three layers in this package:

* **data**, `data.py` and `site_data.py`: loading, enrichment, filtering. No
  rendering.
* **charts**, this package: one function per chart, taking a frame and drawing
  it. A chart knows nothing about which view calls it, so the same chart can
  appear on both the Reporting and the Operations side and cannot disagree with
  itself.
* **views**, the modules beside this package: they choose which charts to show,
  in which order, and write the prose around them.

Charts moved here when a second view wanted one and had to import it from
whichever view happened to own it, `pipeline` importing from `surveys` for the
per-year bars was the case that made the layering worth having.
"""
