#ifdef USE_XNNPACK

#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Common.h>

namespace at {
namespace native {
namespace xnnpack {
namespace {

void hardswish_impl(const Tensor& input, const Tensor& output) {
  using namespace internal;

  xnn_operator_t hardswish_op{};
  const xnn_status create_status = xnn_create_hardswish_nc_f32(
      1, // channels
      1, // input stride
      1, // output stride
      0, // flags
      &hardswish_op);

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_hardswish_nc_f32 failed!");

  Operator hardswish_scoped_op(hardswish_op);

  const xnn_status setup_status = xnn_setup_hardswish_nc_f32(
      hardswish_op,
      input.numel(), // Batch
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      caffe2::pthreadpool_()); // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status, "xnn_setup_hardswish_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      hardswish_op,
      caffe2::pthreadpool_()); // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status, "xnn_run_operator failed!");
}

} // namespace

bool use_hardswish(const Tensor& input) {
  return xnnpack::internal::available() && (1 <= input.ndimension()) &&
      (input.device().is_cpu()) && (kFloat == input.scalar_type()) &&
      !input.requires_grad() && true;
}

void hardswish_out(Tensor& input, Tensor& result) {
  Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
      input, input.suggest_memory_format());

  // Don't need to allocate output if input is contiguous & already padded
  if (result.data_ptr() == padded_input.data_ptr()) {
    hardswish_impl(input, result);
  } else {
    Tensor output = mobile::empty_with_tail_padding(
        padded_input.sizes(),
        padded_input.options().dtype(),
        input.suggest_memory_format(),
        padded_input.names());
    hardswish_impl(padded_input, output);
    result.copy_(output);
  }
}
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
