"""
torchrun --standalone --nproc-per-node 2 test_module_load.py
"""
import torch
import torch.nn as nn
from torchao.dtypes import NF4Tensor

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

device_mesh = init_device_mesh("cuda", (min(2, torch.cuda.device_count()),))
rank = device_mesh.get_rank()
placements = [Shard(0)]

torch.manual_seed(42)
with torch.device('meta'):
    # TODO: Add a buffer to m later
    m = nn.Linear(4, 4, bias=False)
    nf4_weight = NF4Tensor.from_tensor(m.weight.detach(), 2, 2)

    # Emulate the FSDP parameter sharding logic
    assert nf4_weight.size(0) % device_mesh.size() == 0, (
        "Assumes evenly divisible dim-0 for simplicity but got: "
        f"{nf4_weight.size()} {device_mesh}"
    )
    if torch.distributed.get_rank() == 0:
        print(f"before chunk: {nf4_weight.n_blocks}")
    chunks = torch.chunk(nf4_weight, device_mesh.size())
    sharded_nf4_weight = chunks[device_mesh.get_rank()]
    if torch.distributed.get_rank() == 0:
        print(f"after chunk: {sharded_nf4_weight.n_blocks}")
    exit()
    m.weight = torch.nn.Parameter(
        DTensor(
            sharded_nf4_weight,
            device_mesh,
            placements,
            shape=nf4_weight.shape,  # global shape
            dtype=nf4_weight.dtype,
            requires_grad=nf4_weight.requires_grad,
            stride=nf4_weight.stride(),  # global stride
        )
    )

state_dict = {}
weight = torch.randn(4, 4, dtype=torch.bfloat16)
ref_weight = weight.detach().clone()

# Quantize -> chunk
# ref_nf4_weight = NF4Tensor.from_tensor(ref_weight, 2, 2)
# ref_sharded_nf4_weight = torch.chunk(ref_nf4_weight, device_mesh.size())[rank]

# Chunk -> quantize
ref_sharded_weight = torch.chunk(ref_weight, device_mesh.size())[rank]
ref_sharded_nf4_weight = NF4Tensor.from_tensor(ref_sharded_weight, 2, 2)

# TODO: quantize -> chunk vs. chunk -> quantize is not commutative currently.
# The numerics differ slightly! This may be related to the fact that
# scaler_mean is not chunked.

# Given module with DTensor(NF4Tensor) on meta and state dict with
# DTensor(Tensor) on GPU
# - Since self and other are both DTensors, DTensor.module_load calls
#   redistribute.
# - Since self and other are both DTensors with the same device mesh and
#   placements, redistribute is a no-op.
# - DTensor.module_load recursively calls module_load on the local tensors,
#   which runs NF4Tensor.module_load with self: NF4Tensor self and
#   other: Tensor. This calls NF4Tensor.from_tensor(other, ...) to quantize.
# Note that the chunk came from FSDP sharding when initializing the module on
# meta device and not from module_load.
state_dict['weight'] = distribute_tensor(weight, device_mesh, placements)
if rank == 0:
    print(f"[Rank {rank}][before][bf16] {state_dict['weight']}")  # bf16, not quantized
    print(f"[Rank {rank}][before][nf4] {ref_sharded_nf4_weight}")  # nf4, quantized

m.load_state_dict(state_dict, assign=False)
if rank == 0:
    print(f"[Rank {rank}][after] {m.weight.shape} {m.weight.device} local: {m.weight._local_tensor.shape} {m.weight._local_tensor.device}\n{m.weight}")
