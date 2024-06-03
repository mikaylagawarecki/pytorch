#include "miniz.h"
#include <iostream>
#include <pthread.h>
#include <time.h>

#include "caffe2/serialize/crc_alt.h"

extern "C" {
// See: miniz.h
#if defined(USE_EXTERNAL_MZCRC)

typedef struct {
    mz_uint8* ptr;
    size_t buf_len;
    mz_ulong crc;
} ThreadArgs;

void* thread_crc_32(void *arg) {
  ThreadArgs *thread_args = (ThreadArgs*)arg;
  auto z = crc32_fast(thread_args->ptr, thread_args->buf_len, thread_args->crc);
  thread_args->crc = z;
  return (void*)1;
};

mz_ulong mz_crc32(mz_ulong crc, const mz_uint8* ptr, size_t buf_len) {
  clock_t t;
  t = clock();
  // auto out = crc32_fast(ptr, buf_len, crc);
  size_t buf_len_half = buf_len / 2;
  mz_uint8* ptr_half_1 = const_cast<mz_uint8*>(ptr);
  mz_uint8* ptr_half_2 = const_cast<mz_uint8*>(ptr) + buf_len_half;
  pthread_t thread_id1, thread_id2;
  int ret1, ret2;
  // Create the first thread
  ThreadArgs args1 = {
      .ptr = ptr_half_1,
      .buf_len = buf_len_half,
      .crc = 0
  };
  ret1 = pthread_create(&thread_id1, NULL, thread_crc_32, (void*)&args1);
  if (ret1 != 0) {
      printf("Error creating thread 1\n");
      return 1;
  }
  // Create the second thread
  ThreadArgs args2 = {
      .ptr = ptr_half_2,
      .buf_len = buf_len - buf_len_half,
      .crc = 0
  };
  ret2 = pthread_create(&thread_id2, NULL, thread_crc_32, (void*)&args2);
  if (ret2 != 0) {
      printf("Error creating thread 2\n");
      return 1;
  }
  // Wait for both threads to complete
  pthread_join(thread_id1, NULL);
  pthread_join(thread_id2, NULL);
  // Combine the two partial CRC32 results
  auto out = crc32_combine((mz_ulong)((&args1)->crc), (mz_ulong)((&args2)->crc), buf_len - buf_len_half);
  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
  // printf("fun() took %f seconds to execute \n", time_taken);
  return out;
  // size_t buf_len_half = buf_len / 2;
  // mz_uint8* ptr_half_1 = const_cast<mz_uint8*>(ptr);
  // mz_uint8* ptr_half_2 = const_cast<mz_uint8*>(ptr) + buf_len_half;
  // pthread_t thread_id1, thread_id2;
  // int ret1, ret2;
  // // Create the first thread
  // ThreadArgs args1 = {
  //     .ptr = ptr_half_1,
  //     .buf_len = buf_len_half,
  //     .crc = 0
  // };
  // ret1 = pthread_create(&thread_id1, NULL, thread_crc_32, (void*)&args1);
  // if (ret1 != 0) {
  //     printf("Error creating thread 1\n");
  //     return 1;
  // }
  // // Create the second thread
  // ThreadArgs args2 = {
  //     .ptr = ptr_half_2,
  //     .buf_len = buf_len - buf_len_half,
  //     .crc = 0
  // };
  // ret2 = pthread_create(&thread_id2, NULL, thread_crc_32, (void*)&args2);
  // if (ret2 != 0) {
  //     printf("Error creating thread 2\n");
  //     return 1;
  // }
  // // Wait for both threads to complete
  // pthread_join(thread_id1, NULL);
  // pthread_join(thread_id2, NULL);
  // // Combine the two partial CRC32 results
  // auto z = crc32_combine((mz_ulong)((&args1)->crc), (mz_ulong)((&args2)->crc), buf_len - buf_len_half);
  // return z;
};
#endif
}
