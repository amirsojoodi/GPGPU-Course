#define OPERATION(X)   ((sin(X))/1319+(cos(X))/1317+(cos(X+13))*(sin(X-13)))
#define OPERATION_I(X) ((X)/((X)+1010)+(X)/1319+(X)*((X)-13))

__kernel void
vector_operation(__global const int *src,
                 __global       int *trgt
                 )
{
	int gidx = get_global_id(0);

	float r = src[gidx];
	trgt[gidx] = (int)OPERATION(r);
}
