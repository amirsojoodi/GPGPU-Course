#include "main.hxx"

int main(int argc, char* argv[])
{
  using namespace std;

  ios_base::sync_with_stdio(false);

  if (argc != 5) {
    cerr << R"(Correct way to execute this program is:
./multiCq platform_number data_size workgroup_size command_queue_count
For example:\n ./multiCq 0 10000 512 4 )" << endl;
    return 1;
  }

  int platform_number = atoi(argv[1]);
  int n_elem = atoi(argv[2]);
  int local_workg_size = atoi(argv[3]);
  int cq_count = atoi(argv[4]); // cq == command_queue

  vector<int32_t> h_data(n_elem); // host, data
  vector<int32_t> h_output(n_elem); // host, output
  vector<int32_t> d_output(n_elem); // device, output

  // fill A with random doubles
  uniform_int_distribution<> dis(0, RANDOM_MAX);
  generate(h_data.begin(), h_data.end(), bind(dis, RandGen()));

  // copy(A.begin(), A.end(), ostream_iterator<double>(cout, " "));  // print all elements
  auto ts_serial = chrono::high_resolution_clock::now();
  transform(h_data.begin(), h_data.end(), h_output.begin(), OPERATION);  // cpp = GOD
  auto tf_serial = chrono::high_resolution_clock::now();


  // Initialize OpenCL
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  assert(platform_number < (int)platforms.size());
  auto platform = platforms[platform_number]; // here you can select between Intel, AMD or Nvidia
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  auto device = devices.front(); // here you can select between different Accelerators
  auto context = cl::Context(device);

  // Read the program source
  cl::Program program = cl::Program(context, ReadTextFile("./kernel.cl"), CL_FALSE);
  try {
    program.build(BUILD_FLAGS.c_str());
    } catch (const cl::Error& e) {
    if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
      cerr << "[ERROR] Building program Failed, log:\n";
      auto name     = device.getInfo<CL_DEVICE_NAME>();
      auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      cerr << "Build log for " << name << ":\n" << buildLog << endl;
      return 2;
    } else
      throw e;
  }

  // Select kernel
  auto kernel = cl::Kernel(program, "vector_operation");

  // Cerate the device memory buffers
  auto buf_src = cl::Buffer
    (context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, h_data.size() * sizeof(h_data[0]));
  auto buf_target = cl::Buffer
    (context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, d_output.size() * sizeof(d_output[0]));

  // Set the kernel arguments
  kernel.setArg(0, buf_src);
  kernel.setArg(1, buf_target);

  vector<cl::CommandQueue> queues(cq_count);
  generate(queues.begin(), queues.end(), [&](){return cl::CommandQueue(context, device);}); // cpp = 2xGOD

  auto ts_ocl = chrono::high_resolution_clock::now();

  // Compute execution info
  int cq_elem_count = n_elem / cq_count;
  int cq_elem_size = cq_elem_count * sizeof(h_data[0]);
  cl::NDRange global(cq_elem_count);
  cl::NDRange local(local_workg_size);
  for (int i = 0; i < cq_count; ++i) {  // Perform async device calls
    int offset = i * cq_elem_count;
    int offset_with_size = i * cq_elem_size;

    cl::NDRange offset_ndrange(offset);
    vector<cl::Event> ndrange_deps(1), read_deps(1);

    // 3rd(offset) param is on device, 4th(amount) and 5th(start void ptr) are on host
    queues[i].enqueueWriteBuffer
      (buf_src, CL_FALSE, offset_with_size, cq_elem_size, h_data.data() + offset, nullptr, ndrange_deps.data());
    queues[i].enqueueNDRangeKernel
      (kernel, offset_ndrange, global, local, &ndrange_deps, read_deps.data());
    queues[i].enqueueReadBuffer
      (buf_target, CL_FALSE, offset_with_size, cq_elem_size, d_output.data() + offset, &read_deps, nullptr);
  }
  for_each(queues.begin(), queues.end(), [&](auto& q){q.finish();});

  auto tf_ocl = chrono::high_resolution_clock::now();


  double serial_time = chrono::duration<double, milli>(tf_serial - ts_serial).count(),
         ocl_time    = chrono::duration<double, milli>(tf_ocl - ts_ocl).count();
  cout << fixed << showpoint;
  cout.precision(6);
  cout << "[INFO] Serial time:\t"   << serial_time << "ms"
       << "\n[INFO] OpenCL time:\t" << ocl_time    << "ms" <<  '\n';

#ifdef TEST
  auto location = mismatch(h_output.begin(), h_output.end(), d_output.begin());
  if (location.first == h_output.end()) {
    cout.precision(2);
    cout << "[INFO] Test PASS! No mismatch found!\n" << "[INFO] Achived speedup of "
         << serial_time / ocl_time << "x" << endl;
  }
  else
    cout << "[FAIL]  There is a mismatch at location " << (location.first - h_output.begin())
         << "\n\twhere HOST contains " << *location.first << " and DEVICE contains "
         << *location.second << endl;
#endif

  return 0;
}
