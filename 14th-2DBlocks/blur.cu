#include "blur.h"

__global__ void blur_kernel(int *img_out, int *img_in, int height, int width)
{

  int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (tid_x < width - 1 && tid_y < height - 1 && tid_x > 0 && tid_y > 0)
  {
    int sum = 0;
    sum += img_in[(tid_y - 1) * width + tid_x - 1];
    sum += img_in[(tid_y - 1) * width + tid_x];
    sum += img_in[(tid_y - 1) * width + tid_x + 1];
    sum += img_in[tid_y * width + tid_x - 1];
    sum += img_in[tid_y * width + tid_x];
    sum += img_in[tid_y * width + tid_x + 1];
    sum += img_in[(tid_y + 1) * width + tid_x - 1];
    sum += img_in[(tid_y + 1) * width + tid_x];
    sum += img_in[(tid_y + 1) * width + tid_x + 1];
    img_out[tid_y * width + tid_x] = sum / 9;
  }
  else if (tid_x < height && tid_y < width)
  {
  }
}

void sequential_blur(int *img_out, int *img_in, int height, int width)
{

  for (int i = 1; i < height - 1; i++)
  {
    for (int j = 1; j < width - 1; j++)
    {
      int sum = 0;
      sum += img_in[(i - 1) * width + j - 1];
      sum += img_in[(i - 1) * width + j];
      sum += img_in[(i - 1) * width + j + 1];
      sum += img_in[(i)*width + j - 1];
      sum += img_in[(i)*width + j];
      sum += img_in[(i)*width + j + 1];
      sum += img_in[(i + 1) * width + j - 1];
      sum += img_in[(i + 1) * width + j];
      sum += img_in[(i + 1) * width + j + 1];
      img_out[i * width + j] = sum / 9;
    }
  }

  return;
}

void resolve_edges(int *img_out, int *img_in, int height, int width)
{

  for (int i = 0; i < height; i++)
  {
    img_out[i * width] = img_in[i * width];
    img_out[(i + 1) * width - 1] = img_in[(i + 1) * width - 1];
  }

  for (int j = 0; j < width; j++)
  {
    img_out[j] = img_in[j];
    img_out[width * (height - 1) + j] = img_out[width * (height - 1) + j];
  }
}

int main(int argc, char *argv[])
{
  double elapsed_time;
  int block_size_x, grid_size_x;
  int block_size_y, grid_size_y;
  int data_size;
  int *data_h, *output_h, *device_output_h;
  int *data_d, *output_d;
  int stream_count;
  // int work_per_thread;

  if (argc != 4)
  {
    printf("Correct way to execute this program is:\n");
    printf("./blur block_size_x block_size_y stream_count\n");
    printf("For example:\n./blur 16 16 4\n");
    return 1;
  }

  data_size = IMAGE_SIZE_X * IMAGE_SIZE_Y;
  block_size_x = atoi(argv[1]);
  block_size_y = atoi(argv[2]);
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

  // Sequential blur operation
  set_clock();
  sequential_blur(output_h, data_h, IMAGE_SIZE_Y, IMAGE_SIZE_X);
  resolve_edges(output_h, data_h, IMAGE_SIZE_Y, IMAGE_SIZE_X);
  elapsed_time = get_elapsed_time();
  printf("-> Naive blur operation time: %.4fms\n", elapsed_time / 1000);

  // CUDA Parallel blur operation

  // TODO: compute grid_size

  set_clock();

  int stream_size = data_size / stream_count;

  grid_size_x = (IMAGE_SIZE_X - 1) / block_size_x + 1;
  grid_size_y = (IMAGE_SIZE_Y - 1) / (block_size_y * stream_count) + 1;
  dim3 grid_dime(grid_size_x, grid_size_y, 1);
  dim3 block_dime(block_size_x, block_size_y, 1);

  for (int i = 0; i < stream_count; i++)
  {

    int stream_pad_size = (i > 0 && i < stream_count - 1) ? IMAGE_SIZE_X * 2 : IMAGE_SIZE_X;
    int stream_bytes_h_to_d = (stream_size + stream_pad_size) * sizeof(data_d[0]);
    int stream_bytes_d_to_h = (stream_size) * sizeof(data_d[0]);

    int offset = i * stream_size;
    int copy_offset = (i > 0) ? i * stream_size - IMAGE_SIZE_X : offset;

    cudaMemcpyAsync(&data_d[copy_offset], &data_h[copy_offset], stream_bytes_h_to_d, cudaMemcpyHostToDevice, streams[i % STREAM_NUMBERS]);

    int stream_y = IMAGE_SIZE_Y / stream_count;
    stream_y += (i < stream_count - 1) ? 1 : 0;
    stream_y += (i > 0) ? 1 : 0;

    blur_kernel<<<grid_dime, block_dime, 0, streams[i % STREAM_NUMBERS]>>>(&output_d[copy_offset], &data_d[copy_offset], stream_y, IMAGE_SIZE_X);

    cudaMemcpyAsync(&device_output_h[offset], &output_d[offset], stream_bytes_d_to_h, cudaMemcpyDeviceToHost, streams[i % STREAM_NUMBERS]);
  }

  CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
  CUDA_CHECK_RETURN(cudaGetLastError());

  resolve_edges(device_output_h, data_h, IMAGE_SIZE_Y, IMAGE_SIZE_X);

  elapsed_time = get_elapsed_time();

  printf("-> CUDA blur operation time: %.4fms\n", elapsed_time / 1000);

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
