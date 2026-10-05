"""Regression test for t_a512082a: concurrent readers of the install lock must not
serialize behind each other, only behind an actual exclusive writer.

Before this fix, `activate_dependencies` (called on every CLI boot to read the
committed venv selection) took the install lock exclusively even when it was only
reading. With several profile gateways / cron pollers launching `hermes` concurrently,
every boot queued behind whichever one got there first, producing 15s-129s wall-clock
variance on trivially fast commands like `hermes kanban list`.
"""
import os
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pm.filesystem import lock_fd


@pytest.mark.skipif(os.name == "nt", reason="shared locks only differ from exclusive on POSIX")
def test_shared_readers_do_not_serialize_behind_each_other():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "test.lock")
    hold = 0.3
    n = 6
    results = {}

    def reader(idx):
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        t0 = time.monotonic()
        assert lock_fd(fd, wait=True, shared=True)
        time.sleep(hold)
        os.close(fd)
        results[idx] = time.monotonic() - t0

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All shared readers overlap: the slowest should finish close to one hold period,
    # not n * hold (which is what exclusive locking produces — see the sibling test).
    assert max(results.values()) < hold * (n / 2)


@pytest.mark.skipif(os.name == "nt", reason="shared locks only differ from exclusive on POSIX")
def test_exclusive_writer_still_blocks_shared_readers():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "test.lock")
    results = {}

    def writer():
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        assert lock_fd(fd, wait=True)  # exclusive
        time.sleep(0.3)
        os.close(fd)
        results["writer_done_at"] = time.monotonic()

    def reader():
        time.sleep(0.05)  # let the writer acquire first
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        assert lock_fd(fd, wait=True, shared=True)
        results["reader_acquired_at"] = time.monotonic()
        os.close(fd)

    wt = threading.Thread(target=writer)
    rt = threading.Thread(target=reader)
    wt.start()
    rt.start()
    wt.join()
    rt.join()

    # The reader must not acquire before the writer released.
    assert results["reader_acquired_at"] >= results["writer_done_at"]
