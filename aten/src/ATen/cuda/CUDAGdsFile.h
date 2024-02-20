#pragma once

#include <cufile.h>
#include <c10/util/string_view.h>

// FIXME: remove torch.h include
// #include <torch/torch.h>
#include <ATen/Tensor.h>

namespace at::cuda {
  class TORCH_CUDA_CPP_API GDSFile {
    public:
    GDSFile();
    GDSFile(c10::string_view filename, c10::string_view mode);
    ~GDSFile();

    void open(c10::string_view filename, c10::string_view mode);
    void close();

    // FIXME: add file offset (or should these be handled by f.seek(offset) in python)?
    // FIXME: these APIs should take metadata (sizes, dtype, etc. rather than tensor)
    // FIXME: Add API that loads to/from storage as well
    void load_data(const at::Tensor& tensor);
    void save_data(const at::Tensor& tensor);
    void load_data_no_gds(const at::Tensor& tensor);
    void save_data_no_gds(const at::Tensor& tensor);

    private:
    std::string filename;
    std::string mode;

    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;

    int fd = -1;
    bool is_open = false;
    bool maybe_register = true;
  };
} // namespace at::cuda
