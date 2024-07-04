#pragma once

#include <cufile.h>
#include <c10/util/string_view.h>

// FIXME: remove torch.h include
// #include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/Storage.h>

namespace at::cuda {
  class TORCH_CUDA_CPP_API GDSFile {
    public:
    GDSFile();
    GDSFile(c10::string_view filename, c10::string_view mode);
    ~GDSFile();

    void open(c10::string_view filename, c10::string_view mode);
    void close();

    // FIXME: these APIs should take metadata (sizes, dtype, etc. rather than tensor)
    void load_tensor(const at::Tensor& tensor, off_t offset);
    void load_tensors(const std::vector<at::Tensor>& tensors, const std::vector<off_t> offsets);
    void save_tensor(const at::Tensor& tensor, off_t offset);
    void save_tensors(const std::vector<at::Tensor>& tensors, const std::vector<off_t> offsets);
    void load_tensor_no_gds(const at::Tensor& tensor, off_t offset);
    void save_tensor_no_gds(const at::Tensor& tensor, off_t offset);

    void load_storage(const at::Storage& storage, off_t offset);
    void save_storage(const at::Storage& storage, off_t offset);
    void save_storages(const std::vector<at::Storage>& storages, const std::vector<off_t> offsets);
    void load_storage_no_gds(const at::Storage& storage);
    void save_storage_no_gds(const at::Storage& storage);

    void register_buffer(const at::Tensor& tensor);
    void register_buffer(const at::Storage& storage);
    void deregister_buffer(const at::Tensor& tensor);
    void deregister_buffer(const at::Storage& storage);


    private:
    std::string filename;
    std::string mode;

    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;

    CUfileBatchHandle_t cf_batch_handle;

    int fd = -1;
    bool is_open = false;
    bool maybe_register = true;
  };
} // namespace at::cuda
