void kernel simple_add(global const float* A, global const float* B, global float* C)
{
	C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];
}

kernel void sum_reduce(global float* arr, local float* partial_sums, global float* output) // partial_sums must be at least of length get_local_size(0)
{
	int lid = get_local_id(0);
	int group_size = get_local_size(0);
	partial_sums[lid] = arr[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j=2; j/2<group_size; j*=2)
	{
		if ((lid & (j-1)) == 0 && (lid+j/2)<group_size)
			partial_sums[lid] += partial_sums[lid+j/2];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid == 0)
		output[get_group_id(0)] = partial_sums[0];
}