"""BIIGLE request bodies carry the field names BIIGLE validates today.

BIIGLE renamed request fields when reports and enums moved into its core
(2026-10): shape_id -> shape (bulk annotations), type_id -> type (reports),
visibility_id -> visibility (label trees). Every pipeline call failed with
"The <field> field is required" until fixed. These tests pin the names, checked
against biigle/core app/Http/Requests on 2026-11-09; when BIIGLE changes them
again the fix is to send the new name alongside the old, as here.
"""

from unittest.mock import MagicMock

import pytest

from spyfish.biigle.biigle_handler import BiigleHandler


@pytest.fixture
def handler():
    h = BiigleHandler.__new__(BiigleHandler)  # no credentials, no network
    h.api = MagicMock()
    h.api.post.return_value.json.return_value = {"id": 1}
    return h


def test_report_request_sends_type(handler):
    handler.create_report("volumes", 31584, 3)
    body = handler.api.post.call_args.kwargs["json"]
    assert body["type"] == 3, "BIIGLE core validates `type`"
    assert body["type_id"] == 3, "kept for servers before the rename"


def test_bulk_annotations_send_shape(handler):
    handler.api.post.return_value.json.return_value = [{"id": 9}]
    ann = {
        "image_id": 1,
        "shape": 5,
        "shape_id": 5,
        "points": [0, 0, 2, 0, 2, 2, 0, 2],
        "label_id": 7,
        "confidence": 0.5,
    }
    handler.upload_image_annotations(123, [ann])
    sent = handler.api.post.call_args.args[0], handler.api.post.call_args.kwargs["json"]
    assert sent[0] == "image-annotations" and sent[1][0]["shape"] == 5


def test_pipeline_annotation_payload_has_shape():
    """upload_frames builds the bulk payload itself; it must carry `shape`."""
    import inspect

    from spyfish.biigle import upload_frames

    src = inspect.getsource(upload_frames.upload_coco_annotations_to_biigle)
    assert '"shape": 5' in src
