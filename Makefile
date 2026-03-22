PYTHON     ?= python
PIPELINE   = $(PYTHON) run_pipeline.py
DB_ADMIN   = $(PYTHON) -m spyfish.database.set_status

# Optional overrides — pass on the command line, e.g. make db-show DROP_ID=KSF_...
DROP_ID        ?=
STATUS         ?=
SAMPLING_START ?=
SAMPLING_END   ?=
VIDEO_PATH     ?=

# Build --sampling-start / --sampling-end flags only when provided
_SS  = $(if $(SAMPLING_START),--sampling-start $(SAMPLING_START),)
_SE  = $(if $(SAMPLING_END),--sampling-end $(SAMPLING_END),)
_VP  = $(if $(VIDEO_PATH),--video-path $(VIDEO_PATH),)

.PHONY: help \
        run ingest check-arrivals ml \
        zooniverse-clips zooniverse-images zooniverse-sync \
        biigle-upload biigle-sync retrain \
        set-targets \
        run-test run-no-upload \
        set-targets-ml set-targets-ml-biigle \
        ml-biigle biigle-upload-sync biigle-sync-retrain \
        zooniverse expert-review \
        db-show db-set db-create

# ─── Default ────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Individual steps"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make run                     Run all pipeline steps"
	@echo "  make run-test                Run all steps in test mode"
	@echo "  make run-no-upload           Run all steps, skip S3 uploads"
	@echo "  make ingest                  Step 1:  metadata ingestion"
	@echo "  make check-arrivals          Check S3 for new video arrivals"
	@echo "  make ml                      Steps 2+3: ML inference + post-processing"
	@echo "  make zooniverse-clips        Step 4:  Zooniverse clip extraction"
	@echo "  make zooniverse-images       Step 5:  Zooniverse image extraction"
	@echo "  make zooniverse-sync         Step 5b: Zooniverse volunteer sync-back"
	@echo "  make biigle-upload           Step 6:  Biigle frame extraction + upload"
	@echo "  make biigle-sync             Step 7:  Biigle annotation sync"
	@echo "  make retrain                 Step 8:  Model retraining"
	@echo "  make set-targets             Bulk set pipeline stages from targets CSV"
	@echo ""
	@echo "  Combinations"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make set-targets-ml          Set targets then run ML"
	@echo "  make set-targets-ml-biigle   Set targets, ML, then straight to Biigle (skip Zooniverse)"
	@echo "  make ml-biigle               ML + Biigle upload (biigle-direct path, skip Zooniverse)"
	@echo "  make zooniverse              Full Zooniverse path: clips + images + sync"
	@echo "  make expert-review           Biigle upload + sync (when volumes are ready)"
	@echo "  make biigle-sync-retrain     Sync expert annotations then retrain model"
	@echo ""
	@echo "  Database admin"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make db-show   DROP_ID=<id>                      Print record"
	@echo "  make db-set    DROP_ID=<id> STATUS=<status>      Update existing record"
	@echo "                 [SAMPLING_START=<n>] [SAMPLING_END=<n>] [VIDEO_PATH=<p>]"
	@echo "  make db-create DROP_ID=<id> STATUS=<status>      Create new record"
	@echo "                 [SAMPLING_START=<n>] [SAMPLING_END=<n>] [VIDEO_PATH=<p>]"
	@echo ""

# ─── Pipeline steps ─────────────────────────────────────────────────────────

run:
	$(PIPELINE)

run-test:
	$(PIPELINE) --test-run

run-no-upload:
	$(PIPELINE) --no-upload

ingest:
	$(PIPELINE) --ingest

check-arrivals:
	$(PIPELINE) --check-arrivals

ml:
	$(PIPELINE) --ml

zooniverse-clips:
	$(PIPELINE) --zooniverse-clips

zooniverse-images:
	$(PIPELINE) --zooniverse-images

zooniverse-sync:
	$(PIPELINE) --zooniverse-sync

biigle-upload:
	$(PIPELINE) --biigle-upload

biigle-sync:
	$(PIPELINE) --biigle-sync

retrain:
	$(PIPELINE) --retrain

set-targets:
	$(PIPELINE) --set-targets

# ─── Combinations ────────────────────────────────────────────────────────────

# Reset drop targets then immediately run ML on them
set-targets-ml:
	$(PIPELINE) --set-targets --ml

# Reset targets, run ML, then go straight to Biigle (skips Zooniverse entirely)
set-targets-ml-biigle:
	$(PIPELINE) --set-targets --ml --biigle-upload

# Run ML then upload to Biigle directly (biigle-direct path — skips Zooniverse)
ml-biigle:
	$(PIPELINE) --ml --biigle-upload

# Full Zooniverse path: extract clips, extract images, sync volunteer annotations back
zooniverse:
	$(PIPELINE) --zooniverse-clips --zooniverse-images --zooniverse-sync

# Biigle upload + sync annotations back (use when Biigle volumes are ready for review)
expert-review:
	$(PIPELINE) --biigle-upload --biigle-sync

# Sync completed expert annotations from Biigle then trigger model retraining
biigle-sync-retrain:
	$(PIPELINE) --biigle-sync --retrain

# ─── Database admin ──────────────────────────────────────────────────────────

db-show:
	@test -n "$(DROP_ID)" || (echo "Usage: make db-show DROP_ID=<drop_id>" && exit 1)
	$(DB_ADMIN) $(DROP_ID)

db-set:
	@test -n "$(DROP_ID)" || (echo "Usage: make db-set DROP_ID=<drop_id> STATUS=<status>" && exit 1)
	@test -n "$(STATUS)"  || (echo "Usage: make db-set DROP_ID=<drop_id> STATUS=<status>" && exit 1)
	$(DB_ADMIN) $(DROP_ID) $(STATUS) $(_SS) $(_SE) $(_VP) --show

db-create:
	@test -n "$(DROP_ID)" || (echo "Usage: make db-create DROP_ID=<drop_id> STATUS=<status>" && exit 1)
	@test -n "$(STATUS)"  || (echo "Usage: make db-create DROP_ID=<drop_id> STATUS=<status>" && exit 1)
	$(DB_ADMIN) $(DROP_ID) $(STATUS) $(_SS) $(_SE) $(_VP) --create --show
