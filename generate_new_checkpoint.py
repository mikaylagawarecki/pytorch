'''
To generate 70B
    torchrun --standalone --nproc-per-node 8 generate_new_checkpoint.py -s 70

To generate 8B:
    python generate_new_checkpoint.py -s 8

This script expects the llama-70B checkpoints to exist
at `ckpt_dir_map[rank]/70B-consolidated.0{rank}.pth`. See below for `ckpt_dir_map`.

Run this script to repopulate llama-70B checkpoints with random values. 
This is necessary to benchmark `torch.load` time since I don't have sudo access to clear RAM cache.
'''
import torch
import os
import argparse

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

def reset_70B(nvme_per_rank):
    if nvme_per_rank:
        ckpt_dir_map = {0: '/mnt/nvme3/',
                        1: '/mnt/nvme5/'}
        assert local_rank == 0 or local_rank == 1
        fake_local_rank = 2 if local_rank == 0 else 3
    else:
        ckpt_dir_map = {0: '/mnt/nvme0/',
                        1: '/mnt/nvme0/',
                        2: '/mnt/nvme3/',
                        3: '/mnt/nvme3/',
                        4: '/mnt/nvme6/',
                        5: '/mnt/nvme6/',
                        6: '/mnt/nvme7/',
                        7: '/mnt/nvme7/'}
        fake_local_rank = local_rank

    ckpt_path = f"{ckpt_dir_map[local_rank]}/70B-consolidated.0{fake_local_rank}.pth"

    sd = torch.load(ckpt_path, map_location=f'cuda:{local_rank}')
    sd_new = dict()

    for k, v in sd.items():
        # device=cuda for faster randn impl
        sd_new[k] = torch.randn_like(sd[k], device=f'cuda:{local_rank}')

    # use_gds=True to prevent pages from being cached in RAM
    torch.save(sd_new, ckpt_path, use_gds=True)

def reset_8B():
    ckpt_path = f"/mnt/nvme0/consolidated.00.pth"

    sd = torch.load(ckpt_path)
    sd_new = dict()

    for k, v in sd.items():
        # device=cuda for faster randn impl
        sd_new[k] = torch.randn_like(sd[k], device=f'cuda:{local_rank}')

    # use_gds=True to prevent pages from being cached in RAM
    torch.save(sd_new, ckpt_path, use_gds=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example command-line argument parser')
    parser.add_argument('-s', '--size', type=int, help='model to reset')
    parser.add_argument('-n', '--nvme_per_rank', action="store_true", help='use separate nvme for each rank (only works for 2 rank case)')
    args = parser.parse_args()

    if args.size == 8:
        assert not args.nvme_per_rank, "nvme_per_rank only for 70B case"
        reset_8B()
    elif args.size == 70:
        reset_70B(args.nvme_per_rank)
    else:
        raise RuntimeError(f"Got invalid size {args.size}")