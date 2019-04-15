#ifdef OPENCL_2
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#else
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

constexpr int RANDOM_MAX {1000};
std::string OCL2_BUILD_FLAGS {"-cl-std=CL2.0"};
std::string BUILD_FLAGS_BASE {"-cl-mad-enable  -cl-fast-relaxed-math"};
#ifdef OPENCL_2
std::string BUILD_FLAGS{
    std::move(OCL2_BUILD_FLAGS.append(" ").append(BUILD_FLAGS_BASE))};
#else
std::string BUILD_FLAGS{std::move(BUILD_FLAGS_BASE)};
#endif

constexpr auto OPERATION = [](auto X) {
  return sinf(X) / 1319 + cosf(X) / 1317 + cosf(X + 13) * sinf(X - 13);
};
constexpr auto OPERATION_I = [](auto X) {
  return X / (X + 1010) + X / 1319 + X * (X - 13);
};

static inline auto ReadTextFile(const char *s)
{
  std::ifstream mfile(s);
  std::string content((std::istreambuf_iterator<char>(mfile)),
                      (std::istreambuf_iterator<char>()));
  mfile.close();
  return content;
}

static inline auto RandGen()
{
  std::random_device rd;
  std::mt19937 mt(rd());
  return mt;
}
