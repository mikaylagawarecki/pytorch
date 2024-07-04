'''
To bench llama-70B on 8 ranks:
    torchrun --standalone --nproc-per-node 8 bench_llama_70b.py

To bench only one rank's checkpoint:
    torchrun --standalone --nproc-per-node 1 bench_llama_70b.py

Use CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node 2 ...
and uncomment everything that says "For testing 2 ranks with diff NVMe on same switch."
for the testing 2 ranks with diff NVMe on same switch case.

Run this script to benchmark llama-70B save/load time. The configuration in this
script are that each 2 ranks read to/write from a single NVMe drive on the same
PCIe switch. See `ckpt_dir_map` for details.
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

ckpt_dir_map = {0: '/mnt/nvme0/',
                1: '/mnt/nvme0/',
                2: '/mnt/nvme3/',
                3: '/mnt/nvme3/',
                4: '/mnt/nvme6/',
                5: '/mnt/nvme6/',
                6: '/mnt/nvme7/',
                7: '/mnt/nvme7/'}

# For testing 2 ranks with diff NVMe on same switch
# ckpt_dir_map = {0: '/mnt/nvme3/',
#                 1: '/mnt/nvme5/'}

model_parallel_size = None
local_rank = None
ckpt_path = None

def setup():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        global model_parallel_size
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
    global local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    global ckpt_path
    ckpt_path = f"{ckpt_dir_map[local_rank]}/70B-consolidated.0{local_rank}.pth"
    
# For testing 2 ranks with diff NVMe on same switch    
# def fudge_local_rank():
#     fake_local_rank = 2 if local_rank == 0 else 3
#     global ckpt_path
#     ckpt_path = f"{ckpt_dir_map[local_rank]}/70B-consolidated.0{fake_local_rank}.pth"

def init_gpu_direct_storage(local_rank):
    '''
    Calls cuFileDriverOpen in the process, prevents this cold start time from
    being included in benchmarking.
    '''
    start_time = time.time()
    gds_file = torch._C._CudaGdsFileBase(f"{ckpt_dir_map[local_rank]}/bla_{local_rank}.pt", "w")
    print(f"cuFileDriverOpen cold start was {time.time() - start_time}s")
    del gds_file

def sync_ranks():
    torch.distributed.barrier()
    torch.cuda.synchronize()


def baseline_load(ckpt_path, local_rank, mmap):
    sync_ranks()
    start_time = time.time()
    sd = torch.load(ckpt_path, mmap=mmap, map_location=f'cuda:{local_rank}')
    # sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:{local_rank}')
    sync_ranks()
    if local_rank == 0:
        print(f"baseline load time = {time.time() - start_time}")

def baseline_save(ckpt_path, local_rank):
    sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:{local_rank}')
    sync_ranks()
    start_time = time.time()
    torch.save(sd, f"{ckpt_dir_map[local_rank]}save.pth")
    sync_ranks()
    if local_rank == 0:
        print(f"baseline save time = {time.time() - start_time}")


def gpu_direct_load_no_buf_reg(ckpt_path, local_rank):
    sync_ranks()
    start_time = time.time()
    sd = torch.load(ckpt_path, map_location=f'cuda:{local_rank}', use_gds=True)
    sync_ranks()
    if local_rank == 0:
        print(f"gds load time = {time.time() - start_time}")
    # check correctness
    sd_real = torch.load(ckpt_path, map_location=f'cuda:{local_rank}')
    for k in sd.keys():
        assert(torch.equal(sd[k], sd_real[k]))


def gpu_direct_save_no_buf_reg(ckpt_path, local_rank):
    sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:{local_rank}')
    sync_ranks()
    start_time = time.time()
    torch.save(sd, f"{ckpt_dir_map[local_rank]}save_{local_rank}.pth", use_gds=True)
    sync_ranks()
    if local_rank == 0:
        print(f"gds save time = {time.time() - start_time}")
    # check correctness
    sd_saved = torch.load(f"{ckpt_dir_map[local_rank]}save_{local_rank}.pth", mmap=True, map_location=f'cuda:{local_rank}')
    for k, v in sd_saved.items():
        assert(torch.equal(sd[k], sd_saved[k]))


def gpu_direct_load_buf_reg(ckpt_path, local_rank):
    gds_file = torch._C._CudaGdsFileBase(ckpt_path, "r")
    # Get metadata of checkpoint (without actually loading data) (i.e. tensor sizes, device)
    with FakeTensorMode():
        fake_sd = torch.load(ckpt_path, map_location='cpu')
    sd = dict()
    for k, v in fake_sd.items():
        sd[k] = torch.empty(fake_sd[k].size(), dtype=fake_sd[k].dtype, device=f'cuda:{local_rank}')
    sync_ranks()
    start_time = time.time()
    for k in sd.keys():
        assert sd[k].data_ptr() % 4096 == 0
        gds_file.register_buffer(sd[k])
    sync_ranks()
    if local_rank == 0:
        buf_registration_time = time.time() - start_time
        print(f"done with buffer registration in {buf_registration_time}s")
    start_time = time.time()
    for k in sd.keys():
        file_offset = fake_sd[k].untyped_storage().offset
        gds_file.load_tensor(sd[k], file_offset)
    sync_ranks()
    if local_rank == 0:
        print(f"done with gds load in {time.time() - start_time}s")
    # check correctness
    sd_real = torch.load(ckpt_path, map_location=f'cuda:{local_rank}')
    for k in sd.keys():
        assert(torch.equal(sd[k], sd_real[k]))


def gpu_direct_save_buf_reg(ckpt_path, local_rank):
    gds_file = torch._C._CudaGdsFileBase(ckpt_path, "r")
    sd = torch.load(ckpt_path, mmap=True, map_location=f'cuda:{local_rank}')
    sync_ranks()
    start_time = time.time()
    for k in sd.keys():
        assert sd[k].data_ptr() % 4096 == 0
        gds_file.register_buffer(sd[k])
    sync_ranks()
    if local_rank == 0:
        buf_registration_time = time.time() - start_time
        print(f"done with buffer registration in {buf_registration_time}s")
    
    start_time = time.time()
    torch.save(sd, f"{ckpt_dir_map[local_rank]}save_{local_rank}.pth", use_gds=True)
    sync_ranks()
    if local_rank == 0:
        print(f"gds save time = {time.time() - start_time}")
    
    # check correctness
    sd_saved = torch.load(f"{ckpt_dir_map[local_rank]}save_{local_rank}.pth", mmap=True, map_location=f'cuda:{local_rank}')
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

    setup()
    # For testing 2 ranks with diff NVMe on same switch
    # fudge_local_rank()
    assert ckpt_path is not None
    assert local_rank is not None
    if not args.baseline:
        init_gpu_direct_storage(local_rank)

    if args.baseline:
        if args.save:
            baseline_save(ckpt_path, local_rank)
        else:
            assert args.load
            baseline_load(ckpt_path, local_rank, args.mmap)
    else:
        if args.save:
            if args.register_buffers:
                gpu_direct_save_buf_reg(ckpt_path, local_rank)
            else:
                gpu_direct_save_no_buf_reg(ckpt_path, local_rank)
        else:
            assert args.load
            if args.register_buffers:
                gpu_direct_load_buf_reg(ckpt_path, local_rank)
            else:
                gpu_direct_load_no_buf_reg(ckpt_path, local_rank)
    