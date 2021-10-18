#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/ChannelShuffleKernel.h>
#include <ATen/cpu/vec/vec.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void cpu_channel_shuffle(
    Tensor& output,
    const Tensor& input,
    int64_t groups) {
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t channels_per_group = channels / groups;
  int64_t image_size = input.numel() / nbatch / channels;

  // treat input tensor as shape of [n, g, oc, ...]
  // output tensor as shape of [n, oc, g, ...]
  //
  // parallel on dimension of n, oc, g
  using Vec = vec::Vectorized<scalar_t>;
  int64_t inner_size = image_size - (image_size % Vec::size());
  at::parallel_for (0, nbatch * /* oc*g */channels, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oc = 0;
    int64_t g = 0;
    data_index_init(begin, n, nbatch, oc, channels_per_group, g, groups);

    for (int64_t i = begin; i < end; i++) {
      scalar_t* output_ptr = output_data + i * image_size;
      scalar_t* input_ptr = input_data + n * channels * image_size +
          g * channels_per_group * image_size + oc * image_size;

      int64_t d = 0;
      for (; d < inner_size; d += Vec::size()) {
        Vec data_vec = Vec::loadu(input_ptr + d);
        data_vec.store(output_ptr + d);
      }
      for (; d < image_size; d++) {
        output_ptr[d] = input_ptr[d];
      }

      // move on to next output index
      data_index_step(n, nbatch, oc, channels_per_group, g, groups);
    }
  });
}

template <typename scalar_t>
void cpu_channel_shuffle_cl(
    Tensor& output,
    const Tensor& input,
    int64_t groups) {
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t channels_per_group = channels / groups;
  int64_t image_size = input.numel() / nbatch / channels;

  // parallel on dimension of n, h, w
  at::parallel_for(0, nbatch * image_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* output_ptr = output_data + i * channels;
      scalar_t* input_ptr = input_data + i * channels;

      // transpose each channel lane:
      // from [groups, channels_per_group] to [channels_per_group, groups]
      utils::transpose(groups, channels_per_group, input_ptr, channels_per_group, output_ptr, groups);
    }
  });
}

void channel_shuffle_kernel_impl(
    Tensor& output,
    const Tensor& input,
    int64_t groups) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "channel_shuffle", [&] {
        cpu_channel_shuffle<scalar_t>(output, input, groups);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "channel_shuffle_cl", [&] {
        cpu_channel_shuffle_cl<scalar_t>(output, input, groups);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(channel_shuffle_kernel, &channel_shuffle_kernel_impl);

}} // at::native
