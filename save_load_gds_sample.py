import argparse
import os
import psutil
import time

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GDS")
    parser.add_argument('--file', default="/mnt/loopfs/dummy.pth", help="path to file")
    parser.add_argument('--save', default=False, action=argparse.BooleanOptionalAction, help="use gds save")
    parser.add_argument('--load', default=False, action=argparse.BooleanOptionalAction, help="use gds load")
    args = parser.parse_args()
    f = args.file

    message = ("Note that for local filesystem, file must be on filesystem type ext4 or xfs. "
               "See https://github.com/pytorch/pytorch/commit/7374536d9274e7bb739d4c079133127af050b26b "
               "for instructions on how to mount such a filesystem on devgpu.")

    if args.save or args.load:
        assert get_fs_type(f) == "ext4", message

    # each Linear is (10000 * 10000 + 10000) * 4 bytes = ~0.4GB
    # total ~4GB
    torch.manual_seed(5)
    m = nn.ModuleList([nn.Linear(10000, 10000, device='cuda') for i in range(10)])
    sd = m.state_dict()
    torch.cuda.synchronize()
    start_save = time.time()
    torch.save(sd, f, use_gds=args.save)
    print(f"Save time {time.time() - start_save}")

    start_load = time.time()
    sd_loaded = torch.load(f, use_gds=args.load)
    print(f"Load time {time.time() - start_load}")
    for (k1, v1), (k2, v2) in zip(sd.items(), sd_loaded.items()):
        assert k1 == k2
        assert torch.equal(v1, v2)
