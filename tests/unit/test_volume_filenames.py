"""get_volume_images: one request per volume, and never a list with holes.

It used to make one /images/{id} request per image. A 79-volume backfill hit
BIIGLE's rate limit (429), and every failed request was silently dropped, so
reviewed-frames lists were written short (found 2026-10-16 on NeSI).
"""

import pytest
import requests

from spyfish.biigle.biigle_handler import BiigleHandler


class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


class _Api:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url):
        self.calls.append(url)
        r = self.routes(url)
        if isinstance(r, int):
            err = requests.exceptions.HTTPError(f"{r}")
            err.response = _Resp(status=r)
            raise err
        return _Resp(r)


def _handler(routes):
    h = BiigleHandler.__new__(BiigleHandler)  # skip auth/network init
    h.api = _Api(routes)
    return h


def test_one_request_for_the_whole_volume():
    h = _handler(lambda url: {"11": "a.jpg", "12": "b.jpg"})
    imgs = h.get_volume_images(5)
    assert sorted((i["id"], i["filename"]) for i in imgs) == [
        (11, "a.jpg"),
        (12, "b.jpg"),
    ]
    assert h.api.calls == ["volumes/5/filenames"]


def test_rate_limit_raises_instead_of_falling_back():
    h = _handler(lambda url: 429)
    with pytest.raises(requests.exceptions.HTTPError):
        h.get_volume_images(5)
    assert h.api.calls == ["volumes/5/filenames"], "no per-image storm after a 429"


def test_per_image_fallback_refuses_an_incomplete_list():
    def routes(url):
        if url.endswith("/filenames"):
            return 404
        if url.endswith("/files"):
            return [11, 12]
        return 429 if url == "images/12" else {"id": 11, "filename": "a.jpg"}

    with pytest.raises(RuntimeError, match="incomplete"):
        _handler(routes).get_volume_images(5)
