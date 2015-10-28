#ifndef Reduce_cuh
#define Reduce_cuh

#include "AtomicOP.cuh"

template<class T, class Operator, size_t BlockSize>
__global__
void _reduce(const T* arr, size_t n, T* res, const Operator& op, T identity) {
	__shared__ volatile T s_arr[BlockSize];

	size_t local_id = threadIdx.x;
	size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t offset = blockDim.x * gridDim.x;

	s_arr[local_id] = identity;
	while (global_id < n) {
		s_arr[local_id] = op(s_arr[local_id], arr[global_id]);
		global_id += offset;
	}
	__syncthreads();

	if (BlockSize >= 1024) {
		if (local_id < 512)
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 512]);
		__syncthreads();
	}
	if (BlockSize >= 512) {
		if (local_id < 256)
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 256]);
		__syncthreads();
	}
	if (BlockSize >= 256) {
		if (local_id < 128)
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 128]);
		__syncthreads();
	}
	if (BlockSize >= 128) {
		if (local_id < 64)
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 64]);
		__syncthreads();
	}

	if (local_id < 32) {
		if (BlockSize >= 64) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 32]);
		}
		if (BlockSize >= 32) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 16]);
		}
		if (BlockSize >= 16) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 8]);
		}
		if (BlockSize >= 8) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 4]);
		}
		if (BlockSize >= 4) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 2]);
		}
		if (BlockSize >= 2) {
			s_arr[local_id] = op(s_arr[local_id], s_arr[local_id + 1]);
		}
	}

	if (local_id == 0) {
		atomicOP(res, s_arr[0], op);
	}
}

class ReduceGPU {
public:
	static const size_t ELEMS_PER_THREAD = 2048;
	static const size_t BLOCK_SIZE = 128;

	template<class T, class Operator>
	static T reduce(const T* arr, size_t n, const Operator& op) {
		T res = op.identity;
		do {
			cudaError status = cudaSuccess;

			size_t block = BLOCK_SIZE;
			size_t grid = n / (ELEMS_PER_THREAD * BLOCK_SIZE);

			T* d_res = 0;
			status = cudaMalloc(&d_res, sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_res, &res, sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			T* d_arr = 0;
			status = cudaMalloc(&d_arr, n * sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			_reduce<T, Operator, BLOCK_SIZE><<<grid, block>>>(d_arr, n, d_res, op, op.identity);
			
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