# UoA Mussel Insights app

A separate Streamlit app for the University of Auckland mussel-farm BUV dataset
(Underwood & Jeffs, four Hauraki Gulf farms, 18-21 Jan 2022). It is a
collaborator's data, not DOC's, so it does not live in the DOC app
(`app/🐟_Spyfish_Data_Tools.py`) and the DOC app does not reference anything here.
It sits under `in_progress/` with the other unfinished, data-dependent tooling;
it imports the shared `app/` modules by name, so `app.py` puts both folders on the path.
Background: `claude_docs/uoa_design.md`.

## Run

From `app/`, so `app/.streamlit/secrets.toml` resolves:

    cd app && streamlit run ../in_progress/uoa/app.py

## Files

| File | What it is |
|---|---|
| `app.py` | Entrypoint. Puts `app/` on `sys.path`, registers the page below and the shared support page `app/pages/0_🆘_Support.py` |
| `mussel_insights.py` | The dashboard page (farm / industry views, ML performance, ML vs Expert), gated on a password |
| `data.py` | Shared data layer: paths, dataset constants, loaders, comparison metrics |
| `ml_vs_expert.py` | Interval-level ML vs expert comparison, rendered inside Mussel Insights |

Modules here import each other as a package (`from uoa.data import ...`) and
import the shared app modules (`theme`, `utils`, `support`, `ecology_data`)
by their top-level names.

## Secrets (`app/.streamlit/secrets.toml`)

- `TEST_DASHBOARD_PASSWORD`: the page's password gate.
- `[UOA_SITE_COORDS]`: survey positions for the site map. Optional; without
  it the map says it has no positions.

## Data (not in git)

- `hold/uoa/Annotations_CLEANED_v2.csv`: expert annotations.
- `hold/uoa/**/*_raw.csv`: per-chapter ML detections
  (`{deployment}_v{chapter}_{model}_raw.csv`).
- `hold/uoa/uoa_frames/<volume>/`: MaxN frame JPEGs plus
  `deployment_maxn_manifest.csv`, rendered on NeSI by
  `scripts/wip/render_deployment_maxn_frames.py`. Optional.
