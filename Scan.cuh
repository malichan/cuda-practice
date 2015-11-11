#ifndef Scan_cuh
#define Scan_cuh

template<class T, class Operator>
__global__
void _scan_hillis_steele_1(T* arr, size_t n, T* part, const Operator& op, T identity) {
	extern __shared__ T s_arr[];

	size_t local_id = threadIdx.x;
	size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (global_id < n) {
		s_arr[local_id] = arr[global_id];
	} else {
		s_arr[local_id] = identity;
	}
	__syncthreads();

	T* in = &s_arr[0];
	T* out = &s_arr[blockDim.x];
	for (size_t offset = 1; offset < blockDim.x; offset <<= 1) {
		if (local_id >= offset) {
			out[local_id] = op(in[local_id], in[local_id - offset]);
		} else {
			out[local_id] = in[local_id];
		}
		T* temp = in;
		in = out;
		out = temp;
		__syncthreads();
	}

	if (global_id < n) {
		arr[global_id] = in[local_id];
	}

	if (local_id == 0) {
		part[blockIdx.x] = in[blockDim.x - 1];
	}
}

template<class T, class Operator>
__global__
void _scan_hillis_steele_2(T* arr, size_t n, T* part, const Operator& op) {
	if (blockIdx.x > 0) {
		size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
		if (global_id < n) {
			arr[global_id] = op(arr[global_id], part[blockIdx.x - 1]);
		}
	}
}

class ScanGPUHillisSteele  {
public:
	static const size_t BLOCK_SIZE = 512;

	template<class T, class Operator>
	static T* inclusive_scan(const T* arr, size_t n, const Operator& op) {
		T* res = new T[n];
		do {
			cudaError status = cudaSuccess;

			T* d_arr = 0;
			status = cudaMalloc(&d_arr, n * sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			status = _inclusive_scan_device(d_arr, n, op);
			if (status != cudaSuccess) break;

			status = cudaMemcpy(res, d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess) break;

			status = cudaFree(d_arr);
			if (status != cudaSuccess) break;
		} while (false);
		return res;
	}

	template<class T, class Operator>
	static T* exclusive_scan(const T* arr, size_t n, const Operator& op) {
		T* res = new T[n];
		do {
			cudaError status = cudaSuccess;

			T* d_arr = 0;
			status = cudaMalloc(&d_arr, n * sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			status = _inclusive_scan_device(d_arr, n, op);
			if (status != cudaSuccess) break;

			res[0] = op.identity;
			status = cudaMemcpy(res + 1, d_arr, (n - 1) * sizeof(T), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess) break;

			status = cudaFree(d_arr);
			if (status != cudaSuccess) break;
		} while (false);
		return res;
	}

private:
	template<class T, class Operator>
	static cudaError _inclusive_scan_device(T* d_arr, size_t n, const Operator& op) {
		cudaError status = cudaSuccess;

		size_t block = BLOCK_SIZE;
		size_t size = BLOCK_SIZE * 2 * sizeof(T);
		size_t grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

		T* d_part = 0;
		status = cudaMalloc(&d_part, grid * sizeof(T));
		if (status != cudaSuccess) return status;

		_scan_hillis_steele_1<<<grid, block, size>>>(d_arr, n, d_part, op, op.identity);

		if (grid > 1) {
			_inclusive_scan_device(d_part, grid, op);

			_scan_hillis_steele_2<<<grid, block>>>(d_arr, n, d_part, op);
		}

		status = cudaFree(d_part);
		if (status != cudaSuccess) return status;

		return status;
	}
};

const size_t LOG_NUM_BANKS = 5;

template<class T, class Operator>
__global__
void _scan_blelloch_1(T* arr, size_t n, T* part, const Operator& op, T identity) {
	extern __shared__ volatile T s_arr[];

	size_t local_id_0 = threadIdx.x;
	size_t local_id_1 = local_id_0 + blockDim.x;
	local_id_0 += local_id_0 >> LOG_NUM_BANKS;
	local_id_1 += local_id_1 >> LOG_NUM_BANKS;
	size_t global_id_0 = threadIdx.x + blockIdx.x * 2 * blockDim.x;
	size_t global_id_1 = global_id_0 + blockDim.x;

	s_arr[local_id_0] = global_id_0 < n ? arr[global_id_0] : identity;
	s_arr[local_id_1] = global_id_1 < n ? arr[global_id_1] : identity;

	size_t offset, threads;

	for (offset = 1, threads = blockDim.x; threads > 0; offset <<= 1, threads >>= 1) {
		__syncthreads();
		if (threadIdx.x < threads) {
			size_t dst = (threadIdx.x + 1) * offset * 2 - 1;
			size_t src = dst - offset;
			dst += dst >> LOG_NUM_BANKS;
			src += src >> LOG_NUM_BANKS;
			s_arr[dst] = op(s_arr[dst], s_arr[src]);
		}
	}

	if (threadIdx.x == 0) {
		size_t last = blockDim.x * 2 - 1;
		last += last >> LOG_NUM_BANKS;
		part[blockIdx.x] = s_arr[last];
		s_arr[last] = identity;
	}

	for (offset = blockDim.x, threads = 1; offset > 0; offset >>= 1, threads <<= 1) {
		__syncthreads();
		if (threadIdx.x < threads) {
			size_t dst = (threadIdx.x + 1) * offset * 2 - 1;
			size_t src = dst - offset;
			dst += dst >> LOG_NUM_BANKS;
			src += src >> LOG_NUM_BANKS;
			T temp = s_arr[dst];
			s_arr[dst] = op(s_arr[dst], s_arr[src]);
			s_arr[src] = temp;
		}
	}

	__syncthreads();
	if (global_id_0 < n) {
		arr[global_id_0] = s_arr[local_id_0];
	}
	if (global_id_1 < n) {
		arr[global_id_1] = s_arr[local_id_1];
	}
}

template<class T, class Operator>
__global__
void _scan_blelloch_2(T* arr, size_t n, T* part, const Operator& op) {
	if (blockIdx.x > 0) {
		size_t global_id_0 = threadIdx.x + blockIdx.x * 2 * blockDim.x;
		size_t global_id_1 = global_id_0 + blockDim.x;

		if (global_id_0 < n) {
			arr[global_id_0] = op(arr[global_id_0], part[blockIdx.x]);
		}
		if (global_id_1 < n) {
			arr[global_id_1] = op(arr[global_id_1], part[blockIdx.x]);
		}
	}	
}

class ScanGPUBlelloch {
public:
	static const size_t BLOCK_SIZE = 512;

	// template<class T, class Operator>
	// static T* inclusive_scan(const T* arr, size_t n, const Operator& op) {
	// }

	template<class T, class Operator>
	static T* exclusive_scan(const T* arr, size_t n, const Operator& op) {
		T* res = new T[n];
		do {
			cudaError status = cudaSuccess;

			T* d_arr = 0;
			status = cudaMalloc(&d_arr, n * sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			status = _exclusive_scan_device(d_arr, n, op);
			if (status != cudaSuccess) break;

			status = cudaMemcpy(res, d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess) break;

			status = cudaFree(d_arr);
			if (status != cudaSuccess) break;
		} while (false);
		return res;
	}

private:
	template<class T, class Operator>
	static cudaError _exclusive_scan_device(T* d_arr, size_t n, const Operator& op) {
		cudaError status = cudaSuccess;

		size_t block = BLOCK_SIZE;
		size_t size = (BLOCK_SIZE * 2 + ((BLOCK_SIZE * 2) >> LOG_NUM_BANKS)) * sizeof(T);
		size_t grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

		T* d_part = 0;
		status = cudaMalloc(&d_part, grid * sizeof(T));
		if (status != cudaSuccess) return status;

		_scan_blelloch_1<<<grid, block, size>>>(d_arr, n, d_part, op, op.identity);

		if (grid > 1) {
			_exclusive_scan_device(d_part, grid, op);

			_scan_blelloch_2<<<grid, block>>>(d_arr, n, d_part, op);
		}

		status = cudaFree(d_part);
		if (status != cudaSuccess) return status;

		return status;
	}
};

template<class T, class Operator>
__device__
T _scan_efficient_warp(volatile T* s_arr, size_t local_id, size_t lane_id, const Operator& op, T identity) {
	if (lane_id >= 1) {
		s_arr[local_id] = op(s_arr[local_id], s_arr[local_id - 1]);
	}
	if (lane_id >= 2) {
		s_arr[local_id] = op(s_arr[local_id], s_arr[local_id - 2]);
	}
	if (lane_id >= 4) {
		s_arr[local_id] = op(s_arr[local_id], s_arr[local_id - 4]);
	}
	if (lane_id >= 8) {
		s_arr[local_id] = op(s_arr[local_id], s_arr[local_id - 8]);
	}
	if (lane_id >= 16) {
		s_arr[local_id] = op(s_arr[local_id], s_arr[local_id - 16]);
	}

	return s_arr[local_id];
}

template<class T, class Operator>
__global__
void _scan_efficient_1(T* arr, size_t n, T* part, const Operator& op, T identity) {
	extern __shared__ volatile T s_arr[];

	size_t local_id = threadIdx.x;
	size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;

	s_arr[local_id] = global_id < n ? arr[global_id] : identity;

	size_t lane_id = local_id & 31;
	size_t warp_id = local_id >> 5;

	T val = _scan_efficient_warp(s_arr, local_id, lane_id, op, identity);
	__syncthreads();

	if (lane_id == 31)
		s_arr[warp_id] = s_arr[local_id];
	__syncthreads();

	if (warp_id == 0)
		_scan_efficient_warp(s_arr, local_id, lane_id, op, identity);
	__syncthreads();

	if (warp_id > 0)
		val = op(val, s_arr[warp_id - 1]);
	__syncthreads();

	s_arr[local_id] = val;
	__syncthreads();

	if (global_id < n) {
		arr[global_id] = local_id > 0 ? s_arr[local_id - 1] : identity;
	}

	if (local_id == 0) {
		part[blockIdx.x] = s_arr[blockDim.x - 1];
	}
}

template<class T, class Operator>
__global__
void _scan_efficient_2(T* arr, size_t n, T* part, const Operator& op) {
	if (blockIdx.x > 0) {
		size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (global_id < n) {
			arr[global_id] = op(arr[global_id], part[blockIdx.x]);
		}
	}
}

class ScanGPUEfficient {
public:
	static const size_t BLOCK_SIZE = 512;

	// template<class T, class Operator>
	// static T* inclusive_scan(const T* arr, size_t n, const Operator& op) {
	// }

	template<class T, class Operator>
	static T* exclusive_scan(const T* arr, size_t n, const Operator& op) {
		T* res = new T[n];
		do {
			cudaError status = cudaSuccess;

			T* d_arr = 0;
			status = cudaMalloc(&d_arr, n * sizeof(T));
			if (status != cudaSuccess) break;
			status = cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyHostToDevice);
			if (status != cudaSuccess) break;

			status = _exclusive_scan_device(d_arr, n, op);
			if (status != cudaSuccess) break;

			status = cudaMemcpy(res, d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess) break;

			status = cudaFree(d_arr);
			if (status != cudaSuccess) break;
		} while (false);
		return res;
	}

private:
	template<class T, class Operator>
	static cudaError _exclusive_scan_device(T* d_arr, size_t n, const Operator& op) {
		cudaError status = cudaSuccess;

		size_t block = BLOCK_SIZE;
		size_t size = BLOCK_SIZE * sizeof(T);
		size_t grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

		T* d_part = 0;
		status = cudaMalloc(&d_part, grid * sizeof(T));
		if (status != cudaSuccess) return status;

		_scan_efficient_1<<<grid, block, size>>>(d_arr, n, d_part, op, op.identity);

		if (grid > 1) {
			_exclusive_scan_device(d_part, grid, op);

			_scan_efficient_2<<<grid, block>>>(d_arr, n, d_part, op);
		}

		status = cudaFree(d_part);
		if (status != cudaSuccess) return status;

		return status;
	}
};

class ScanCPU {
public:
	template<class T, class Operator>
	static T* inclusive_scan(const T* arr, size_t n, const Operator& op) {
		T sum = op.identity;
		T* res = new T[n];
		for (size_t i = 0; i < n; ++i) {
			sum = op(sum, arr[i]);
			res[i] = sum;
		}
		return res;
	}

	template<class T, class Operator>
	static T* exclusive_scan(const T* arr, size_t n, const Operator& op) {
		T sum = op.identity;
		T* res = new T[n];
		for (size_t i = 0; i < n; ++i) {
			res[i] = sum;
			sum = op(sum, arr[i]);
		}
		return res;
	}
};

#endif