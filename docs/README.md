# docs/

`index.html` is the operations runbook: a single self-contained page listing
everything in the toolkit that can be run, grouped by use case.

It is plain HTML with no build step and no external assets beyond Google Fonts,
so GitHub Pages can serve it directly:

> Settings → Pages → Source: *Deploy from a branch* → Branch: `main`, folder: `/docs`

The page then lives at `https://wildlifeai.github.io/spyfish-aotearoa-toolkit/`.

To edit, open `index.html` and change the content in place. The content is
derived from `design_doc.md`, `CLAUDE.md`, `README.md` and the CLI help of the
entry points, so when a flag changes, update both.
