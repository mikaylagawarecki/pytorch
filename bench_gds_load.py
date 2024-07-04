import os
import timeit
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import namedtuple

os.environ["CUFILE_ENV_PATH_JSON"] = "/home/mg1998/pytorch/cufile.json"
LOWER_BOUND = 10
UPPER_BOUND= 29
NUM_ITERS = 10
BufRegistrationConfig = namedtuple('BufRegistrationConfig',
                                   ['register_buffer', 'include_buf_registration_in_time'])


def run_benchmark_torch_load():
    '''
    For each size from 2 ** LOWER_BOUND to 2 ** UPPER_BOUND, create NUM_ITERS+1
    random normal tensors of size and save to /mnt/nvme0/size-{*}, with 1 as warmup.
    '''
    torch.manual_seed(5)
    sizes = [2 ** i for i in range(LOWER_BOUND, UPPER_BOUND)]
    results = []
    for size in sizes:
        torch.cuda.empty_cache()
        xs = []
        ys = [torch.randn(size, device = "cuda") for i in range(NUM_ITERS + 1)]
        
        for i in range(NUM_ITERS+1):
            torch.save(ys[i], f"/mnt/nvme0/{size}-{i}.data")

        # warmup CUDACachingAllocator
        # torch.cuda.synchronize()
        x = xs.append(torch.load(f"/mnt/nvme0/{size}-0.data"))

        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        
        for i in range(NUM_ITERS):
            xs.append(torch.load(f"/mnt/nvme0/{size}-{i+1}.data"))
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        results.append((end_time - start_time) / NUM_ITERS)
        print(f"torch.load: size = {size}, {(end_time - start_time) / NUM_ITERS}")
        assert(all([torch.equal(x, y) for x, y in zip(xs, ys)]))
        
    return results

def run_benchmark(func,
                  open_mode,
                  batched=False,
                  buf_registration: BufRegistrationConfig = BufRegistrationConfig(True, False)):
    '''
    For each size from 2 ** LOWER_BOUND to 2 ** UPPER_BOUND, create NUM_ITERS+1
    random normal tensors of size and save to /mnt/nvme0/size-{*} with func, 1 is used as warmup.
    '''
    torch.manual_seed(5)
    sizes = [2 ** i for i in range(LOWER_BOUND, UPPER_BOUND)]
    assert not (not buf_registration.register_buffer and buf_registration.include_buf_registration_in_time)
    preregister_buffer = buf_registration.register_buffer and not buf_registration.include_buf_registration_in_time
    register_buffer_during_benchmark = buf_registration.register_buffer and buf_registration.include_buf_registration_in_time
    results = []
    for size in sizes:
        torch.cuda.empty_cache()
        # Create tensors for GPUDirect to load to
        xs = [torch.empty(size, device = "cuda") for i in range(NUM_ITERS+1)]
        ys = [torch.randn(size, device = "cuda") for i in range(NUM_ITERS+1)]
        f = torch._C._CudaGdsFileBase(f"/mnt/nvme0/{size}.data", "w")
        # since min num_bytes is 2**10 * 4 (for torch.float32) = 4096, 4KB alignment is guaranteed for GDS
        offsets = list(range(0, ys[0].nbytes * (NUM_ITERS + 1), ys[0].nbytes))
        for i in range(NUM_ITERS+1):
            f.save_tensor(ys[i], offsets[i])

        f1 = torch._C._CudaGdsFileBase(f"/mnt/nvme0/{size}.data", open_mode)
        if 'n' not in open_mode and preregister_buffer:
            [f1.register_buffer(x) for x in xs]
        

        # warmup
        torch.cuda.synchronize()
        func(xs[-1], f1, offsets[-1])

        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        if batched:
            func(xs, f1, offsets)
        else:
            for i in range(NUM_ITERS):
                if register_buffer_during_benchmark:
                        f1.register_buffer(xs[i])
                        func(xs[i], f1, offsets[i])
                        f1.deregister_buffer(xs[i])
                else:
                    func(xs[i], f1, offsets[i])
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        results.append((end_time - start_time)/NUM_ITERS)
        print(f"{func.__name__}: size = {size * 4}, {(end_time - start_time)/NUM_ITERS}")
        if 'n' not in open_mode and preregister_buffer:
            [f1.deregister_buffer(x) for x in xs]
        del f1
        assert(all([torch.equal(x, y) for x, y in zip(xs, ys)]))
    return results

def load_data_yes_gds(tensor, f, offset):
    f.load_tensor(tensor, offset)

def load_data_no_gds(tensor, f, offset):
    f.load_tensor_no_gds(tensor, offset)

def load_data_batched_gds(tensors, f, offsets):
    f.load_tensors(tensors, offsets)

if __name__ == '__main__':
    # Step (1): Run this to get the base data and save it
    data = dict()
    data['size (bytes)'] = [4 * 2 ** i for i in range(LOWER_BOUND, UPPER_BOUND)]
    data['torch.load'] = run_benchmark_torch_load()
    data['gds (4 threads, request_parallelism 4)'] = run_benchmark(load_data_yes_gds, "r")
    data['gds (4 threads, request_parallelism 4), no reg_buf'] = run_benchmark(load_data_yes_gds, "r", buf_registration=BufRegistrationConfig(False, False))
    data['gds (4 threads, request_parallelism 4), reg_buf in bench'] = run_benchmark(load_data_yes_gds, "r", buf_registration=BufRegistrationConfig(True, True))
    data['no gds'] = run_benchmark(load_data_no_gds, "rn")
    df = pd.DataFrame(data)
    df['size (GB)'] = df['size (bytes)']/1e9
    df.to_json('data_load.json', orient='records')

    # Step(2): Uncomment and run this (with appropriate settings in cufile.json to get the other rows)
    # df = pd.read_json('data_load.json')
    # df['gds (no threadpool)'] = run_benchmark(load_data_yes_gds, "r")
    # df.to_json('data_load.json', orient='records')

    # Step (3): Uncomment and run this to get the plots
    # df = pd.read_json('data_load.json')
    # # df = df[:-12]
    # df.plot(x='size (GB)',
    #         y=['torch.load',
    #              'gds (4 threads, request_parallelism 4)',
    #              'gds (2 threads, request_parallelism 2)',
    #              'gds (4 threads, request_parallelism 4), no reg_buf',
    #              'gds (4 threads, request_parallelism 4), reg_buf in bench',
    #              'gds (no threadpool)',
    #              'no gds'],
    #         color=['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink'], marker='.')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.xlabel('size (GB)')
    # plt.ylabel('Load time (s)')
    # plt.savefig('load_time.png')


    # # For batched case
    # # df = pd.read_json('data.json')
    # # df = df[0:12]
    # # df['gds (batched)'] = run_benchmark(load_data_batched_gds, "r")
    print(df)