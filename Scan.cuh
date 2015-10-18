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

			_scan_hillis_steele_2<<<grid, block, size>>>(d_arr, n, d_part, op);
		}

		status = cudaFree(d_part);
		if (status != cudaSuccess) return status;

		return status;
	}
};

class ScanGPUBlelloch {
public:
	static const size_t BLOCK_SIZE = 512;

	// template<class T, class Operator>
	// static T* inclusive_scan(const T* arr, size_t n, const Operator& op) {
	// }

	// template<class T, class Operator>
	// static T* exclusive_scan(const T* arr, size_t n, const Operator& op) {
	// }
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