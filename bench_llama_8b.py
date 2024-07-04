'''
To load llama-8B on rank 0:
    python bench_llama_8b.py
'''

import torch
import time
import os
import argparse

from pathlib import Path
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from torch._subclasses.fake_tensor import FakeTensorMode

os.environ["CUFILE_ENV_PATH_JSON"] = "/home/mg1998/pytorch/cufile.json"

ckpt_path = f"/mnt/nvme0/consolidated.00.pth"

def init_gpu_direct_storage():
    '''
    Calls cuFileDriverOpen in the process, prevents this cold start time from
    being included in benchmarking.
    '''
    start_time = time.time()
    gds_file = torch._C._CudaGdsFileBase(f"/mnt/nvme0/bla_0.pt", "w")
    print(f"cuFileDriverOpen cold start was {time.time() - start_time}s")
    del gds_file

def baseline_load(ckpt_path, mmap):
    start_time = time.time()
    sd = torch.load(ckpt_path, mmap=mmap, map_location=f'cuda:0')
    # torch.cuda.synchronize()
    print(f"baseline load time = {time.time() - start_time}")

def baseline_save(ckpt_path):
    sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:0')
    torch.cuda.synchronize()
    start_time = time.time()
    torch.save(sd, f"/mnt/nvme0/save.pth")
    torch.cuda.synchronize()
    print(f"baseline save time = {time.time() - start_time}")


def gpu_direct_load_no_buf_reg(ckpt_path):
    start_time = time.time()
    sd = torch.load(ckpt_path, map_location=f'cuda:0', use_gds=True)
    torch.cuda.synchronize()
    print(f"gds load time = {time.time() - start_time}")
    # check correctness
    sd_real = torch.load(ckpt_path, map_location=f'cuda:0')
    for k in sd.keys():
        assert(torch.equal(sd[k], sd_real[k]))


def gpu_direct_save_no_buf_reg(ckpt_path):
    sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:0')
    torch.cuda.synchronize()
    start_time = time.time()
    torch.save(sd, f"/mnt/nvme0/save_0.pth", use_gds=True)
    torch.cuda.synchronize()
    print(f"gds save time = {time.time() - start_time}")
    # check correctness
    sd_saved = torch.load(f"/mnt/nvme0/save_0.pth", mmap=True, map_location=f'cuda:0')
    for k, v in sd_saved.items():
        assert(torch.equal(sd[k], sd_saved[k]))


def gpu_direct_load_buf_reg(ckpt_path):
    gds_file = torch._C._CudaGdsFileBase(ckpt_path, "r")
    # Get metadata of checkpoint (without actually loading data) (i.e. tensor sizes, device)
    with FakeTensorMode():
        fake_sd = torch.load(ckpt_path, map_location='cpu')
    sd = dict()
    for k, v in fake_sd.items():
        sd[k] = torch.empty(fake_sd[k].size(), dtype=fake_sd[k].dtype, device=f'cuda:0')
    torch.cuda.synchronize()
    start_time = time.time()
    for k in sd.keys():
        gds_file.register_buffer(sd[k])
    torch.cuda.synchronize()
    buf_registration_time = time.time() - start_time
    print(f"done with buffer registration in {buf_registration_time}s")
    start_time = time.time()
    for k in sd.keys():
        file_offset = fake_sd[k].untyped_storage().offset
        gds_file.load_tensor(sd[k], file_offset)
    torch.cuda.synchronize()
    print(f"done with gds load in {time.time() - start_time}s")
    # check correctness
    sd_real = torch.load(ckpt_path, map_location=f'cuda:0')
    for k in sd.keys():
        assert(torch.equal(sd[k], sd_real[k]))


def gpu_direct_save_buf_reg(ckpt_path):
    gds_file = torch._C._CudaGdsFileBase(ckpt_path, "r")
    sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:0')
    torch.cuda.synchronize()
    start_time = time.time()
    for k in sd.keys():
        gds_file.register_buffer(sd[k])
    torch.cuda.synchronize()
    buf_registration_time = time.time() - start_time
    print(f"done with buffer registration in {buf_registration_time}s")
    
    start_time = time.time()
    torch.save(sd, f"/mnt/nvme0/save_0.pth", use_gds=True)
    torch.cuda.synchronize()
    print(f"gds save time = {time.time() - start_time}")
    
    # check correctness
    sd_saved = torch.load(f"/mnt/nvme0/save_0.pth", mmap=True, map_location=f'cuda:0')
    for k, v in sd_saved.items():
        assert(torch.equal(sd[k], sd_saved[k]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example command-line argument parser')
    parser.add_argument('-l', '--load', action="store_true", help='benchmark loading')
    parser.add_argument('-s', '--save', action="store_true", help='benchmark saving')
    parser.add_argument('-b', '--baseline', action="store_true", help='run baseline')
    parser.add_argument('-r', '--register_buffers', action="store_true", help='run cuFileBufRegister on each buffer')
    parser.add_argument('-m', '--mmap', action="store_true", help='run baseline torch.load with mmap=True')
    args = parser.parse_args()
    assert not (args.save and args.load), "Expected one of -s or -l but got both"
    assert not (args.register_buffers and args.baseline), "Got both -r and -b but cannot cuFileBufRegister for baseline"
    assert not ((not (args.baseline and args.load)) and args.mmap), "Got -m but mmap is only for load baseline"

    if not args.baseline:
        init_gpu_direct_storage()

    if args.baseline:
        if args.save:
            baseline_save(ckpt_path)
        else:
            assert args.load
            baseline_load(ckpt_path, args.mmap)
    else:
        if args.save:
            if args.register_buffers:
                gpu_direct_save_buf_reg(ckpt_path)
            else:
                gpu_direct_save_no_buf_reg(ckpt_path)
        else:
            assert args.load
            if args.register_buffers:
                gpu_direct_load_buf_reg(ckpt_path)
            else:
                gpu_direct_load_no_buf_reg(ckpt_path)