#include "main.hxx"

constexpr bool USE_NAIVE{false};

int main(int argc, char *argv[]) {
  using namespace std;

  ios_base::sync_with_stdio(false);

  if (argc != 5) {
    cerr << "Correct way to execute this program is:" << endl
         << "./vectorOp platform_number data_size workgroup_size "
            "work_per_workitem"
         << endl
         << "For example:\n./vectorOp 0 10000 512 4 " << endl;
    return 1;
  }

  int platform_number = atoi(argv[1]);
  int n_elem = atoi(argv[2]);
  int local_work_size = atoi(argv[3]);
  int work_per_workitem = atoi(argv[4]);

  vector<int32_t> h_data(n_elem);
  vector<int32_t> h_output(n_elem);
  vector<int32_t> d_output(n_elem);

  // fill A with random ints
  uniform_int_distribution<> dis(0, RANDOM_MAX);
  generate(h_data.begin(), h_data.end(), bind(dis, RandGen()));

  /// Serial Execution
  auto ts_serial = chrono::high_resolution_clock::now();
  transform(h_data.begin(), h_data.end(), h_output.begin(), OPERATION_I);
  auto tf_serial = chrono::high_resolution_clock::now();

  // Initialize OpenCL
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  assert(platform_number < (int)platforms.size());
  auto platform = platforms[platform_number];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  auto device = devices.front();
  auto context = cl::Context(device);
  auto queue = cl::CommandQueue(context, device);

  // Cerate the device memory buffers
  auto buf_src = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            h_data.size() * sizeof(h_data[0]));
  auto buf_trgt = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                             d_output.size() * sizeof(d_output[0]));

  // Read the program source
  cl::Program program =
      cl::Program(context, ReadTextFile("./kernel.cl"), CL_FALSE);
  try {
    program.build(BUILD_FLAGS.c_str());
  } catch (const cl::Error &e) {
    if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
      cerr << "[ERROR] Building program Failed, log:\n";
      auto name = device.getInfo<CL_DEVICE_NAME>();
      auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      cerr << "Build log for " << name << ":\n" << buildLog << endl;
      return 2;
    } else
      throw e;
  }

  // Select kernel implementation
  auto kernel = cl::Kernel(
      program,
      ((const char *[]){
          "op", "op_coalesced"})[1 - USE_NAIVE]); // c11: compound literal

  // Set the kernel arguments
  kernel.setArg(0, buf_src);
  kernel.setArg(1, buf_trgt);
  kernel.setArg(2, work_per_workitem); // useless for op_coalesced kernel
  kernel.setArg(3, n_elem);

  // Set lunch args
  cl::NDRange global(n_elem / work_per_workitem);
  cl::NDRange local(local_work_size);

  queue.finish();

  /// Enqueue kernel for execution
  auto ts_ocl = chrono::high_resolution_clock::now();
  queue.enqueueWriteBuffer(buf_src, CL_TRUE, 0,
                           h_data.size() * sizeof(h_data[0]), h_data.data());
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
  queue.enqueueReadBuffer(buf_trgt, CL_TRUE, 0,
                          d_output.size() * sizeof(d_output[0]),
                          d_output.data());
  queue.finish();
  auto tf_ocl = chrono::high_resolution_clock::now();

  cout << "Serial time:\t"
       << chrono::duration<double, milli>(tf_serial - ts_serial).count() * 10
       << "ms"
       << "\nOpenCL time:\t"
       << chrono::duration<double, milli>(tf_ocl - ts_ocl).count() * 10 << "ms"
       << '\n';

#ifdef TEST
  auto location = mismatch(h_output.begin(), h_output.end(), d_output.begin());
  if (location.first == h_output.end())
    cout << "[PASS] No mismatch found! " << endl;
  else
    cout << "[FAIL]  There is a mismatch at location "
         << (location.first - h_output.begin()) << "\n\twhere HOST contains "
         << *location.first << " and DEVICE contains " << *location.second
         << endl;
#endif

  return 0;
}
