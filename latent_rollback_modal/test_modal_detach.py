from __future__ import annotations

import io
from contextlib import redirect_stdout


class _FakeCall:
    def __init__(self, object_id: str):
        self.object_id = object_id


class _FakeRemoteFn:
    def __init__(self):
        self.spawned = None
        self.called = None

    def spawn(self, args):
        self.spawned = args
        return _FakeCall("fc-123")

    def remote(self, args):
        self.called = args
        return {"exit_code": 0, "files": {}}


def test_launch_detached_uses_spawn_and_prints_call_id():
    from latent_rollback_modal.modal_cli import launch_remote

    fn = _FakeRemoteFn()
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = launch_remote(fn, ["--foo"], detach=True)

    assert exit_code == 0
    assert fn.spawned == ["--foo"]
    assert fn.called is None
    assert "fc-123" in buf.getvalue()


def test_launch_remote_waits_when_not_detached(tmp_path):
    from latent_rollback_modal.modal_cli import launch_remote

    fn = _FakeRemoteFn()
    exit_code = launch_remote(fn, ["--bar"], detach=False, results_root=tmp_path)

    assert exit_code == 0
    assert fn.called == ["--bar"]
    assert fn.spawned is None
