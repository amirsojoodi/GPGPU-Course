#include"blur.h"

__global__ void blur_kernel_shared_memory_revised_II(int *img_out, int *img_in, int height, int width, int pad){

        int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
        int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

        extern __shared__ int img_in_s[]; // size = (blockDim.y + 2) * (blockDim.x + 2)

		int tid_shared = (threadIdx.y+1)*(blockDim.x+pad)+(threadIdx.x+1);

        img_in_s[tid_shared] = img_in[(tid_y+1) * (width+2) + tid_x+1];

        if(threadIdx.y == blockDim.y - 1){ // Last row
                img_in_s[tid_shared + (blockDim.x+pad)] =
                        img_in[(tid_y+2) * (width+2) + tid_x];
                img_in_s[tid_shared + (blockDim.x+pad)] +=
                        img_in[(tid_y+2) * (width+2) + tid_x+1];
                img_in_s[tid_shared + (blockDim.x+pad)] +=
                        img_in[(tid_y+2) * (width+2) + tid_x+2];
        }

        if(threadIdx.y == 0){ // First row
                img_in_s[tid_shared - (blockDim.x+pad)] =
                        img_in[(tid_y) * (width+2) + tid_x];
                img_in_s[tid_shared - (blockDim.x+pad)] +=
                        img_in[(tid_y) * (width+2) + tid_x+1];
                img_in_s[tid_shared - (blockDim.x+pad)] +=
                        img_in[(tid_y) * (width+2) + tid_x+2];
        }

		if(threadIdx.x == 0){ //First Column
				img_in_s[tid_shared - 1] =
			            img_in[(tid_y+1) * (width+2) + tid_x];
		}

		if(threadIdx.x == blockDim.x - 1) { //Last Column
				img_in_s[tid_shared + 1] =
						img_in[(tid_y+1) * (width+2) + tid_x+2];
		}

        __syncthreads();
		
        int sum = img_in_s[tid_shared];
		sum += img_in_s[tid_shared - 1];
		sum += img_in_s[tid_shared + 1];

		__syncthreads();

		img_in_s[tid_shared] = sum;

		__syncthreads();

        sum += img_in_s[tid_shared - (blockDim.x+pad)];
		sum += img_in_s[tid_shared + (blockDim.x+pad)];

        img_out[tid_y * width + tid_x] = sum / 9;
}

__global__ void blur_kernel_shared_memory_revised_I(int *img_out, int *img_in, int height, int width, int pad){
	
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	extern __shared__ int img_in_s[]; // size = (blockDim.y + 2) * (blockDim.x + 2)

	img_in_s[(threadIdx.y)*(blockDim.x+pad)+(threadIdx.x)] = 
		img_in[(tid_y) * (width+2) + tid_x];
	
	if(threadIdx.y == blockDim.y - 1){ // Last row
		img_in_s[(threadIdx.y+1)*(blockDim.x+pad)+(threadIdx.x)] = 
			img_in[(tid_y+1) * (width+2) + tid_x];
		img_in_s[(threadIdx.y+2)*(blockDim.x+pad)+(threadIdx.x)] = 
			img_in[(tid_y+2) * (width+2) + tid_x];

		if(threadIdx.y == blockDim.y - 1){ // Last Thread				
			img_in_s[(threadIdx.y+1)*(blockDim.x+pad)+(threadIdx.x+1)] = 
				img_in[(tid_y+1) * (width+2) + tid_x+1];
			img_in_s[(threadIdx.y+2)*(blockDim.x+pad)+(threadIdx.x+1)] = 
				img_in[(tid_y+2) * (width+2) + tid_x+1];
			img_in_s[(threadIdx.y+1)*(blockDim.x+pad)+(threadIdx.x+2)] = 
				img_in[(tid_y+1) * (width+2) + tid_x+2];
			img_in_s[(threadIdx.y+2)*(blockDim.x+pad)+(threadIdx.x+2)] = 
				img_in[(tid_y+2) * (width+2) + tid_x+2];
		}
	}
	
	if(threadIdx.x == blockDim.x - 1){ // Last column
		img_in_s[(threadIdx.y)*(blockDim.x+pad)+(threadIdx.x+1)] = 
			img_in[(tid_y) * (width+2) + tid_x+1];
		img_in_s[(threadIdx.y)*(blockDim.x+pad)+(threadIdx.x+2)] = 
			img_in[(tid_y) * (width+2) + tid_x+2];
	}

	__syncthreads();

	int sum = 0;
	sum += img_in_s[(threadIdx.y) * (blockDim.x+pad) + threadIdx.x];
	sum += img_in_s[(threadIdx.y) * (blockDim.x+pad) + threadIdx.x+1];
	sum += img_in_s[(threadIdx.y) * (blockDim.x+pad) + threadIdx.x+2];
	sum += img_in_s[(threadIdx.y+1) * (blockDim.x+pad) + threadIdx.x];
	sum += img_in_s[(threadIdx.y+1) * (blockDim.x+pad) + threadIdx.x+1];
	sum += img_in_s[(threadIdx.y+1) * (blockDim.x+pad) + threadIdx.x+2];
	sum += img_in_s[(threadIdx.y+2) * (blockDim.x+pad) + threadIdx.x];
	sum += img_in_s[(threadIdx.y+2) * (blockDim.x+pad) + threadIdx.x+1];
	sum += img_in_s[(threadIdx.y+2) * (blockDim.x+pad) + threadIdx.x+2];
	sum /= 9;

	img_out[tid_y * width + tid_x] = sum;
}

__global__ void blur_kernel(int *img_out, int *img_in, int height, int width){
	
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	int sum = 0;
	sum += img_in[(tid_y) * (width + IMAGE_PAD_SIZE) + tid_x];
	sum += img_in[(tid_y) * (width + IMAGE_PAD_SIZE) + tid_x + 1];
	sum += img_in[(tid_y) * (width + IMAGE_PAD_SIZE) + tid_x + 2];
	sum += img_in[(tid_y + 1) * (width + IMAGE_PAD_SIZE) + tid_x];
	sum += img_in[(tid_y + 1) * (width + IMAGE_PAD_SIZE) + tid_x + 1];
	sum += img_in[(tid_y + 1) * (width + IMAGE_PAD_SIZE) + tid_x + 2];
	sum += img_in[(tid_y + 2) * (width + IMAGE_PAD_SIZE) + tid_x];
	sum += img_in[(tid_y + 2) * (width + IMAGE_PAD_SIZE) + tid_x + 1];
	sum += img_in[(tid_y + 2) * (width + IMAGE_PAD_SIZE) + tid_x + 2];
	img_out[tid_y * width + tid_x] = sum / 9;
}

void sequential_blur(int *img_out, int *img_in, int height, int width){

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			int sum = 0;
			sum += img_in[(i) * (width + IMAGE_PAD_SIZE) + j];
			sum += img_in[(i) * (width + IMAGE_PAD_SIZE) + j + 1];
			sum += img_in[(i) * (width + IMAGE_PAD_SIZE) + j + 2];
			sum += img_in[(i + 1) * (width + IMAGE_PAD_SIZE) + j];
			sum += img_in[(i + 1) * (width + IMAGE_PAD_SIZE) + j + 1];
			sum += img_in[(i + 1) * (width + IMAGE_PAD_SIZE) + j + 2];
			sum += img_in[(i + 2) * (width + IMAGE_PAD_SIZE) + j];
			sum += img_in[(i + 2) * (width + IMAGE_PAD_SIZE) + j + 1];
			sum += img_in[(i + 2) * (width + IMAGE_PAD_SIZE) + j + 2];

			img_out[i * width + j] = sum / 9;
		}
	}
	return;
}

int main(int argc, char *argv[]){

	double elapsed_time;
	int block_size_x, grid_size_x;
	int block_size_y, grid_size_y;
	int input_size;
	int output_size;
	int *input_h, *output_h, *device_output_h;
	int *input_d, *output_d;
	int stream_count;
	// int work_per_thread;

	if(argc != 4){
		printf("Correct way to execute this program is:\n");
		printf("./blur block_size_x block_size_y stream_count\n");
		printf("For example:\n./blur 16 16 4\n");
		return 1;
	}

	input_size = (IMAGE_SIZE_X + IMAGE_PAD_SIZE) * (IMAGE_SIZE_Y + IMAGE_PAD_SIZE);
	output_size = IMAGE_SIZE_X * IMAGE_SIZE_Y;

	block_size_x = atoi(argv[1]);
	block_size_y = atoi(argv[2]);
	stream_count = atoi(argv[3]);
	// work_per_thread = atoi(argv[4]);

	cudaStream_t* streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * STREAM_NUMBERS);

	for(int i = 0; i < STREAM_NUMBERS; i++){
		cudaStreamCreate(&streams[i]);
	}

	// Initialize data on Host
	initialize_data_random_cudaMallocHost(&input_h, input_size);
	initialize_data_zero(&output_h, output_size);
	initialize_data_zero_cudaMallocHost(&device_output_h, output_size);
	
	// Initialize data on Device
	CUDA_CHECK_RETURN(cudaMalloc((void **)&input_d, sizeof(int)*input_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&output_d, sizeof(int)*output_size));
	
	// Perform GPU Warm-up
	CUDA_CHECK_RETURN(cudaMemcpyAsync(input_d, input_h, sizeof(int), cudaMemcpyHostToDevice, streams[0]));

	// Sequential blur operation
	
	set_clock();

	sequential_blur(output_h, input_h, IMAGE_SIZE_Y, IMAGE_SIZE_X);

    elapsed_time = get_elapsed_time();

	printf("-> Naive blur operation time: %.4fms\n", elapsed_time / 1000);

	// CUDA Parallel blur operation

	set_clock();

	int stream_size = output_size / stream_count;

	grid_size_x = (IMAGE_SIZE_X - 1) / block_size_x + 1;
	grid_size_y = (IMAGE_SIZE_Y - 1) / (block_size_y * stream_count) + 1;
	dim3 grid_dime(grid_size_x, grid_size_y, 1);
	dim3 block_dime(block_size_x, block_size_y, 1);

	int pad = IMAGE_PAD_SIZE + 1;
	int shared_memory_size = (block_size_x + pad) * 
		(block_size_y + IMAGE_PAD_SIZE) * sizeof(input_d[0]);

	for(int i = 0; i < stream_count; i++){

		int stream_pad_size = (IMAGE_SIZE_X + IMAGE_PAD_SIZE) * IMAGE_PAD_SIZE + 
			(IMAGE_SIZE_Y / stream_count) * IMAGE_PAD_SIZE;
		int stream_bytes_h_to_d = (stream_size + stream_pad_size) * sizeof(input_d[0]);
		int stream_bytes_d_to_h = (stream_size) * sizeof(input_d[0]);

		int offset = i * stream_size;
		int copy_offset = (i > 0)? i * (IMAGE_SIZE_Y / stream_count) * 
			(IMAGE_SIZE_X + IMAGE_PAD_SIZE): offset;

		cudaMemcpyAsync(&input_d[copy_offset], &input_h[copy_offset], stream_bytes_h_to_d, cudaMemcpyHostToDevice, streams[i % STREAM_NUMBERS]);
		
		int stream_y = IMAGE_SIZE_Y / stream_count;

		blur_kernel_shared_memory_revised_II<<< grid_dime, block_dime, shared_memory_size, streams[i % STREAM_NUMBERS]>>>(&output_d[offset], &input_d[copy_offset], stream_y, IMAGE_SIZE_X, pad);

		cudaMemcpyAsync(&device_output_h[offset], &output_d[offset], stream_bytes_d_to_h, cudaMemcpyDeviceToHost, streams[i % STREAM_NUMBERS]);
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

    elapsed_time = get_elapsed_time();

    printf("-> CUDA blur operation time: %.4fms\n", elapsed_time / 1000);

	#ifdef DEBUG
	int width = IMAGE_SIZE_X + IMAGE_PAD_SIZE;
	printf("img[%d, %d] = %d\n", 0, 0, input_h[0]);
	printf("img[%d, %d] = %d\n", 0, 1, input_h[1]);
	printf("img[%d, %d] = %d\n", 0, 2, input_h[2]);
	printf("img[%d, %d] = %d\n", 1, 0, input_h[width]);
	printf("img[%d, %d] = %d\n", 1, 1, input_h[width+1]);
	printf("img[%d, %d] = %d\n", 1, 2, input_h[width+2]);
	printf("img[%d, %d] = %d\n", 2, 0, input_h[width*2]);
	printf("img[%d, %d] = %d\n", 2, 1, input_h[width*2+1]);
	printf("img[%d, %d] = %d\n", 2, 2, input_h[width*2+2]);
	#endif

    #ifdef  TEST
    validate(output_h, device_output_h, output_size);
    #endif

	for (int i = 0; i < STREAM_NUMBERS; i++){
        cudaStreamDestroy(streams[i]);
    }

	free(output_h);
	free(streams);
	CUDA_CHECK_RETURN(cudaFreeHost(input_h));
	CUDA_CHECK_RETURN(cudaFreeHost(device_output_h));
	CUDA_CHECK_RETURN(cudaFree(output_d));
	CUDA_CHECK_RETURN(cudaFree(input_d));

	return 0;
}
