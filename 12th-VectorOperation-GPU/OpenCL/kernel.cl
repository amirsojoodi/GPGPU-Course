#define OPERATION(X)                                                           \
  (sin(X)) / 1319 + (cos(X)) / 1317 + (cos(X + 13)) * (sin(X - 13))
#define OPERATION_I(X) (X) / ((X) + 1010) + (X) / 1319 + (X) * ((X)-13)

__kernel void op(__global const int *src, __global int *trgt,
                 int work_per_workitem, int src_size) {
  int si = work_per_workitem * get_global_id(0);
  int ei = min(si + work_per_workitem, src_size);
  // int lid = get_local_id(0);

  for (int i = si; i < ei; ++i) {
    int r = src[i];
    trgt[i] = OPERATION_I(r);
  }
}

__kernel void op_coalesced(__global const int *src, __global int *trgt,
                           int work_per_workitem, int src_size) {
  int i = get_global_id(0);
  // int lid = get_local_id(0);

  while (i < src_size) {
    int r = src[i];
    trgt[i] = OPERATION_I(r);

    // Jump to next
    i += get_global_size(0);
  }
}