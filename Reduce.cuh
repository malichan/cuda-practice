#ifndef Reduce_cuh
#define Reduce_cuh

#include "AtomicOP.cuh"

template<class T, class Operator>
__global__
void _reduce(const T* arr, size_t n, T* res, const Operator& op, T identity) {
	extern __shared__ T s_arr[];

	size_t local_id = threadIdx.x;
	size_t global_id = threadIdx.x + (blockIdx.x * 2) * blockDim.x;
	size_t offset = blockDim.x;

	if (global_id + offset < n) {
		s_arr[local_id] = arr[global_id] + arr[global_id + offset];
	} else if (global_id < n) {
		s_arr[local_id] = arr[global_id];
	} else {
		s_arr[local_id] = identity;
	}
	__syncthreads();

	for (offset >>= 1; offset > 0; offset >>= 1) {
		if (local_id < offset) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + offset]);
		}
		__syncthreads();
	}

	if (local_id == 0) {
		atomicOP(res, s_arr[0], op);
	}
}

class ReduceGPU {
public:
	static const size_t BLOCK_SIZE = 512;

	template<class T, class Operator>
	static T reduce(const T* arr, size_t n, const Operator& op) {
		T res = op.identity;
		do {
			cudaError status = cudaSuccess;

			size_t block = BLOCK_SIZE;
			size_t size = BLOCK_SIZE * sizeof(T);
			size_t grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

			T* d_arr = 0;
			status = cudaMalloc(&d_arr, n * sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			T* d_res = 0;
			status = cudaMalloc(&d_res, sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_res, &res, sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			_reduce<<<grid, block, size>>>(d_arr, n, d_res, op, op.identity);
			
			status = cudaMemcpy(&res, d_res, sizeof(T), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess) break;

			status = cudaFree(d_arr);
			if (status != cudaSuccess) break;
			status = cudaFree(d_res);
			if (status != cudaSuccess) break;
		} while (false);
		return res;
	}
};

class ReduceCPU {
public:
	template<class T, class Operator>
	static T reduce(const T* arr, size_t n, const Operator& op) {
		T res = op.identity;
		for (size_t i = 0; i < n; ++i) {
			res = op(res, arr[i]);
		}
		return res;
	}
};

#endif