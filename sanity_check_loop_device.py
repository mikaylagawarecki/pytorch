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
    def test_basic(self):
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name

            t = torch.arange(1000, dtype=torch.float32, device="cuda:0")
            t1 = torch.empty(1000, dtype=torch.float32, device="cuda:1")

            g = torch._C._CudaGdsFileBase(filename, "w")

            g.save_data(t)
            del g

            h = torch._C._CudaGdsFileBase(filename, "r")
            h.load_data(t1)

            self.assertEqual(t, t1)
            del h


if __name__ == "__main__":
    run_tests()
