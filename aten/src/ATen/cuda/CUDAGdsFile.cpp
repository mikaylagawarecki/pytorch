#include <ATen/cuda/CUDAGdsFile.h>

// torch
#include <c10/cuda/CUDAGuard.h>

// cuda
#include <cuda_runtime.h>
#include <cufile.h>
#include <chrono>
#include <iostream>

// file io
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>


namespace at::cuda {

#define MAX_BATCH_IOS 128

// POSIX
template <
    class T,
    typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type =
        nullptr>
std::string cuGDSFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(std::strerror(errno));
}

// CUfileError_t
template <
    class T,
    typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type =
        nullptr>
std::string cuGDSFileGetErrorString(T status) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}

GDSFile::GDSFile() : is_open(false) {};

GDSFile::GDSFile(c10::string_view filename, c10::string_view mode) : filename(filename), mode(mode), is_open(false) {
  open(filename, mode);
}

GDSFile::~GDSFile() {
  if (is_open) {
    close();
  }
}

void GDSFile::open(c10::string_view other_filename, c10::string_view other_mode) {
  TORCH_CHECK(is_open == false, "file", filename, "is already open");
  if (!filename.empty()) {
    TORCH_CHECK(other_filename == filename, "file", filename, "is already open with mode", mode);
  }
  if (!mode.empty()) {
    TORCH_CHECK(other_mode == mode, "file", filename, "is already open with mode", mode);
  }

  maybe_register = true;
  // Open the binary file
  if (mode == "r") {
    // for reading
    fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
  } else if (mode == "w") {
    // for writing
    fd = ::open(filename.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0664);
  } else if (mode == "rn") {
    fd = ::open(filename.c_str(), O_RDONLY);
    maybe_register = false;
  } else if (mode == "wn") {
    // for writing
    fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    maybe_register = false;
  } else {
    TORCH_CHECK(false, "only r and w modes are currently supported, but got:", mode);
  }
  TORCH_CHECK(fd >= 0, "fcntl cannot open file: ", filename);

  // Register cuGDSFile handle
  if (maybe_register) {
      memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
      cf_descr.handle.fd = fd;
      cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      status = cuFileHandleRegister(&cf_handle, &cf_descr);
      if (status.err != CU_FILE_SUCCESS) {
        TORCH_CHECK(false, "cuFileHandleRegister failed: ", cuGDSFileGetErrorString(status));
      }
  }
  is_open = true;
}

void GDSFile::close() {
  // Deregister cuGDSFile handle and close the file
  if (is_open) {
      if (maybe_register) {
        cuFileHandleDeregister(cf_handle);
      }
      ::close(fd);
      fd = -1;
  }
  is_open = false;
}

void GDSFile::register_buffer(const at::Tensor& tensor) {
  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufRegister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void GDSFile::register_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufRegister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void GDSFile::deregister_buffer(const at::Tensor& tensor) {
  void* dataPtr = tensor.data_ptr();
  status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufDeregister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void GDSFile::deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufDeregister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void GDSFile::load_tensors(const std::vector<at::Tensor>& tensors, const std::vector<off_t> offsets) {
  unsigned batch_size = tensors.size();
  unsigned nr;
  CUfileIOParams_t io_batch_params[batch_size];
	CUfileIOEvents_t io_batch_events[batch_size];
  int num_completed = 0;

  for(unsigned i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handle;
		io_batch_params[i].u.batch.devPtr_base = tensors[i].data_ptr();
		io_batch_params[i].u.batch.file_offset = offsets[i];
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = tensors[i].nbytes();
		io_batch_params[i].opcode = CUFILE_READ;
	}

	status = cuFileBatchIOSetUp(&cf_batch_handle, batch_size);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed: ", cuGDSFileGetErrorString(status));

  status = cuFileBatchIOSubmit(cf_batch_handle, batch_size, io_batch_params, /*flags=*/0);	
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit failed: ", cuGDSFileGetErrorString(status));

  while (num_completed != batch_size) {
    memset(io_batch_events, 0, sizeof(*io_batch_events));
    nr = batch_size;
    status = cuFileBatchIOGetStatus(cf_batch_handle, batch_size, &nr, io_batch_events, NULL);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus failed: ", cuGDSFileGetErrorString(status));
    num_completed += nr;
    TORCH_WARN("num_completed: ", num_completed);
  }

  cuFileBatchIODestroy(cf_batch_handle);
}

void GDSFile::load_tensor(const at::Tensor& tensor, off_t offset) {
  TORCH_CHECK(mode == "r", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, (void*)dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
}

void GDSFile::load_storage(const at::Storage& storage, off_t offset) {
  TORCH_CHECK(mode == "r", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, (void*)dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
}

// FIXME: For now, the assumption is made that all requests submitted in a batch
// have a consistent
// (1) File
// (2) Mode (either reading or writing)
// This does not use the full generality that the APIs offer but should be
// sufficient for my benchmarking purposes.
void GDSFile::save_tensors(const std::vector<at::Tensor>& tensors, const std::vector<off_t> offsets) {
  unsigned batch_size = tensors.size();
  unsigned nr;
  CUfileIOParams_t io_batch_params[batch_size];
	CUfileIOEvents_t io_batch_events[batch_size];
  int num_completed = 0;

  for(unsigned i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handle;
		io_batch_params[i].u.batch.devPtr_base = tensors[i].data_ptr();
		io_batch_params[i].u.batch.file_offset = offsets[i];
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = tensors[i].nbytes();
		io_batch_params[i].opcode = CUFILE_WRITE;
	}

	status = cuFileBatchIOSetUp(&cf_batch_handle, batch_size);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed: ", cuGDSFileGetErrorString(status));

  status = cuFileBatchIOSubmit(cf_batch_handle, batch_size, io_batch_params, /*flags=*/0);	
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit failed: ", cuGDSFileGetErrorString(status));

  while (num_completed != batch_size) {
    memset(io_batch_events, 0, sizeof(*io_batch_events));
    nr = batch_size;
    status = cuFileBatchIOGetStatus(cf_batch_handle, batch_size, &nr, io_batch_events, NULL);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus failed: ", cuGDSFileGetErrorString(status));
    num_completed += nr;
  }

  cuFileBatchIODestroy(cf_batch_handle);
}

void GDSFile::save_tensor(const at::Tensor& tensor, off_t offset) {
  TORCH_CHECK(mode == "w", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

void GDSFile::save_storage(const at::Storage& storage, off_t offset) {
  TORCH_CHECK(mode == "w", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  // FIXME: check whether storage.mutable_data() is the correct API to call here
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

void GDSFile::save_storages(const std::vector<at::Storage>& tensors, const std::vector<off_t> offsets) {
  unsigned batch_size = tensors.size();
  unsigned nr;
  CUfileIOParams_t io_batch_params[batch_size];
	CUfileIOEvents_t io_batch_events[batch_size];
  int num_completed = 0;

  for(unsigned i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handle;
		io_batch_params[i].u.batch.devPtr_base = tensors[i].mutable_data();
		io_batch_params[i].u.batch.file_offset = offsets[i];
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = tensors[i].nbytes();
		io_batch_params[i].opcode = CUFILE_WRITE;
	}

	status = cuFileBatchIOSetUp(&cf_batch_handle, batch_size);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed: ", cuGDSFileGetErrorString(status));

  status = cuFileBatchIOSubmit(cf_batch_handle, batch_size, io_batch_params, /*flags=*/0);	
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit failed: ", cuGDSFileGetErrorString(status));

  while (num_completed != batch_size) {
    memset(io_batch_events, 0, sizeof(*io_batch_events));
    nr = batch_size;
    status = cuFileBatchIOGetStatus(cf_batch_handle, batch_size, &nr, io_batch_events, NULL);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus failed: ", cuGDSFileGetErrorString(status));
    num_completed += nr;
  }

  cuFileBatchIODestroy(cf_batch_handle);
}


// Just for benchmarking purposes

void GDSFile::load_tensor_no_gds(const at::Tensor& tensor, off_t offset) {
  TORCH_CHECK(mode == "rn", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtrCPU = nullptr;
  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");
  ssize_t nbytes_read = 0;
  off_t load_offset = offset;

  while (nbytes_read !=  nbytes){
    const ssize_t bytes_read = pread(fd, dataPtrCPU + nbytes_read, nbytes - nbytes_read, load_offset);
    TORCH_CHECK(bytes_read == nbytes - nbytes_read || bytes_read == 0 || bytes_read == 2147479552, "fcntl pread failed ", bytes_read, nbytes - bytes_read);
    nbytes_read += bytes_read;
    load_offset += bytes_read;
  }
  
  C10_CUDA_CHECK(cudaMemcpy(dataPtr, dataPtrCPU, nbytes, cudaMemcpyHostToDevice));
  free(dataPtrCPU);

  // while (nbytes_read != nbytes){
  //   // TORCH_WARN("READING");
  //   const ssize_t bytes_read = pread(fd, dataPtrCPU + nbytes_read, nbytes - nbytes_read, load_offset);
  //   TORCH_CHECK(bytes_read == nbytes - nbytes_read || bytes_read == 0 || bytes_read == 2147479552, "fcntl pread failed ", bytes_read, nbytes - nbytes_read);
  //   nbytes_read += bytes_read;
  //   load_offset += bytes_read;
  // }
  // C10_CUDA_CHECK(cudaMemcpy(dataPtr, dataPtrCPU, nbytes, cudaMemcpyHostToDevice));
  // free(dataPtrCPU);
}

void GDSFile::load_storage_no_gds(const at::Storage& storage) {
  TORCH_CHECK(mode == "rn", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtrCPU = nullptr;
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();
  dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");

  const ssize_t nbytes_read = pread(fd, dataPtrCPU, nbytes, 0);
  TORCH_CHECK(nbytes_read == nbytes || nbytes_read == 0, "fcntl pread failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtr, dataPtrCPU, nbytes, cudaMemcpyHostToDevice));
  free(dataPtrCPU);
}

void GDSFile::save_tensor_no_gds(const at::Tensor& tensor, off_t offset) {
  TORCH_CHECK(mode == "wn", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtrCPU = nullptr;
  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();
  dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtrCPU, dataPtr, nbytes, cudaMemcpyDeviceToHost));

  ssize_t nbytes_written = 0;
  off_t write_offset = offset;

  while (nbytes_written !=  nbytes){
    const ssize_t bytes_written = pwrite(fd, dataPtrCPU + nbytes_written, nbytes - nbytes_written, write_offset);
    TORCH_CHECK(bytes_written == nbytes - nbytes_written || bytes_written == 0 || bytes_written == 2147479552, "fcntl pread failed ", bytes_written, nbytes - bytes_written);
    nbytes_written += bytes_written;
    write_offset += bytes_written;
  }
  free(dataPtrCPU);
}

void GDSFile::save_storage_no_gds(const at::Storage& storage) {
  TORCH_CHECK(mode == "wn", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtrCPU = nullptr;
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();
  dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtrCPU, dataPtr, nbytes, cudaMemcpyDeviceToHost));

  const ssize_t nbytes_written = pwrite(fd, dataPtrCPU, nbytes, 0);
  TORCH_CHECK(nbytes_written == nbytes, "fcntl pwrite failed");
  free(dataPtrCPU);
}

} // namespace at::cuda
