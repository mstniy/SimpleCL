kernel void sum_reduce(global float* arr, local float* partial_sums, global float* output, int N) // partial_sums must be at least of length get_local_size(0)
{
	int lid = get_local_id(0);
	int group_size = get_local_size(0);
	if (get_global_id(0) >= N) // Pretend that *arr* was padded with 0s.
		partial_sums[lid] = 0;
	else
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
