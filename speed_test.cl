void kernel increment(global const float* A, global float* B)
{
	B[get_global_id(0)] = A[get_global_id(0)]+1;
}
