#include "Operators.cuh"
#include "Reduce.cuh"
#include "Scan.cuh"

#include <ctime>
#include <cstdlib>
#include <iostream>
using namespace std;

#define TYPE float

int main() {
	srand(time(0));
	size_t n = 100000000;
	TYPE* arr = new TYPE[n];
	for (size_t i = 0; i < n; ++i) {
		arr[i] = static_cast<TYPE>(20.0 * rand() / RAND_MAX - 10.0);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	TYPE reduce_cpu = ReduceCPU::reduce(arr, n, Operators::add<TYPE>());
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float cpu_time = 0.0f;
	cudaEventElapsedTime(&cpu_time, start, stop);

	cout << "Reduce CPU: " << reduce_cpu << endl;
	cout << "Execution time: " << cpu_time << " ms" << endl;

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	TYPE reduce_gpu = ReduceGPU::reduce(arr, n, Operators::add<TYPE>());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpu_time = 0.0f;
	cudaEventElapsedTime(&gpu_time, start, stop);

	cout << "Reduce GPU: " << reduce_gpu << endl;
	cout << "Execution time: " << gpu_time << " ms" << endl;

	// TYPE* inc_scan_cpu = ScanCPU::inclusive_scan(arr, n, Operators::add<TYPE>());
	// cout << "Inclusive Scan CPU: ";
	// for (size_t i = 0; i < n; i += n >> 3)
	// 	cout << inc_scan_cpu[i] << " ";
	// cout << endl;
	// TYPE* inc_scan_gpu = ScanGPUBlelloch::inclusive_scan(arr, n, Operators::add<TYPE>());
	// cout << "Inclusive Scan GPU: ";
	// for (size_t i = 0; i < n; i += n >> 3)
	// 	cout << inc_scan_gpu[i] << " ";
	// cout << endl;

	// delete [] inc_scan_cpu;
	// delete [] inc_scan_gpu;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delete [] arr;
}