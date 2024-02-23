"""
TMPDIR="/mnt/loopfs/" python sanity_check_loop_device.py
"""

import os

import psutil


def get_fs_type(mypath):
    mypath = os.path.realpath(mypath)
    root_type = ""
    for part in psutil.disk_partitions():
        if part.mountpoint == "/":
            root_type = part.fstype
            continue

        if mypath.startswith(part.mountpoint):
            return part.fstype

    return root_type


# Getting the tempdir
import tempfile

assert get_fs_type(tempfile.gettempdir()) == "ext4"
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCudaGdsFileBase(TestCase):
    def test_basic_tensor(self):
        with tempfile.NamedTemporaryFile() as f:
            size = (1000,)
            filename = f.name

            t = torch.randn(size, dtype=torch.float32, device="cuda:0")
            t1 = torch.empty(size, dtype=torch.float32, device="cuda:1")

            g = torch._C._CudaGdsFileBase(filename, "w")

            g.save_tensor(t)
            del g

            h = torch._C._CudaGdsFileBase(filename, "r")
            h.load_tensor(t1)

            self.assertEqual(t, t1)
            del h

    def test_basic_storage(self):
        with tempfile.NamedTemporaryFile() as f:
            size = (3,)
            filename = f.name

            t = torch.randn(size, dtype=torch.float32, device="cuda:0")
            t1 = torch.empty(size, dtype=torch.float32, device="cuda:1")

            g = torch._C._CudaGdsFileBase(filename, "w")

            g.save_storage(t.untyped_storage())
            del g

            h = torch._C._CudaGdsFileBase(filename, "r")
            h.load_storage(t1.untyped_storage())

            self.assertEqual(t, t1)
            del h


if __name__ == "__main__":
    run_tests()
