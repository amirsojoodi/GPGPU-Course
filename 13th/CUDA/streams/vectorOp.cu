#include "vectorOp.h"

__global__ void vector_operation_kernel(int *output, int *data, int size)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < size)
  {
    output[tid] = OPERATION(data[tid]);
  }
}

void sequential_vector_operation(int *output, int *data, int size)
{

  for (int i = 0; i < size; i++)
  {
    output[i] = OPERATION(data[i]);
  }
  return;
}

int main(int argc, char *argv[])
{

  double elapsed_time;
  int block_size, grid_size;
  int data_size;
  int *data_h, *output_h, *device_output_h;
  int *data_d, *output_d;
  int stream_count;
  // int work_per_thread;

  if (argc != 4)
  {
    printf("Correct way to execute this program is:\n");
    printf("./vectorOp data_size block_size stream_count\n");
    printf("For example:\n./vectorOp 10000 512 4\n");
    return 1;
  }

  data_size = atoi(argv[1]);
  block_size = atoi(argv[2]);
  stream_count = atoi(argv[3]);
  // work_per_thread = atoi(argv[4]);

  cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * STREAM_NUMBERS);

  for (int i = 0; i < STREAM_NUMBERS; i++)
  {
    cudaStreamCreate(&streams[i]);
  }

  // Initialize data on Host
  //initialize_data_random(&data_h, data_size);
  initialize_data_random_cudaMallocHost(&data_h, data_size);
  initialize_data_zero(&output_h, data_size);
  //initialize_data_zero(&device_output_h, data_size);
  initialize_data_zero_cudaMallocHost(&device_output_h, data_size);

  // Initialize data on Device
  CUDA_CHECK_RETURN(cudaMalloc((void **)&data_d, sizeof(int) * data_size));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&output_d, sizeof(int) * data_size));

  // Sequential vector operation
  set_clock();
  for (int r = 0; r < REPEAT; r++)
  {
    sequential_vector_operation(output_h, data_h, data_size);
  }
  elapsed_time = get_elapsed_time();
  printf("-> Naive vector operation time: %.4fms\n", elapsed_time / (REPEAT * 1000));

  // CUDA Parallel vector operation

  // TODO: compute grid_size

  set_clock();

  for (int r = 0; r < REPEAT; r++)
  {

    int stream_size = data_size / stream_count;
    int stream_bytes = stream_size * sizeof(data_d[0]);

    grid_size = (stream_size - 1) / block_size + 1;
    dim3 grid_dime(grid_size, 1, 1);
    dim3 block_dime(block_size, 1, 1);
    int offset = 0;

    for (int i = 0; i < stream_count; i++)
    {

      cudaMemcpyAsync(&data_d[offset], &data_h[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]);
      offset += stream_size;
    }
    offset = 0;
    for (int i = 0; i < stream_count; i++)
    {
      vector_operation_kernel<<<grid_dime, block_dime, 0, streams[i]>>>(&output_d[offset], &data_d[offset], stream_size);
      offset += stream_size;
    }
    offset = 0;
    for (int i = 0; i < stream_count; i++)
    {
      cudaMemcpyAsync(&device_output_h[offset], &output_d[offset], stream_bytes, cudaMemcpyDeviceToHost, streams[i]);
      offset += stream_size;
    }

    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN(cudaGetLastError());
  }
  elapsed_time = get_elapsed_time();

  printf("-> CUDA vector operation time: %.4fms\n", elapsed_time / (1000 * REPEAT));

#ifdef TEST
  validate(output_h, device_output_h, data_size);
#endif

  for (int i = 0; i < STREAM_NUMBERS; i++)
  {
    cudaStreamDestroy(streams[i]);
  }

  //free(data_h);
  CUDA_CHECK_RETURN(cudaFreeHost(data_h));
  free(output_h);
  free(streams);
  //free(device_output_h);
  CUDA_CHECK_RETURN(cudaFreeHost(device_output_h));

  CUDA_CHECK_RETURN(cudaFree(output_d));
  CUDA_CHECK_RETURN(cudaFree(data_d));

  return 0;
}
