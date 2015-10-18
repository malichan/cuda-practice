#include "Operators.cuh"
#include "Reduce.cuh"
#include "Scan.cuh"

#include <iostream>
using namespace std;

#define TYPE long long

int main() {
	size_t n = 1000000;
	TYPE* arr = new TYPE[n];
	for (size_t i = 0; i < n; ++i)
		arr[i] = (i + 1);

	TYPE reduce_cpu = ReduceCPU::reduce(arr, n, Operators::add<TYPE>());
	cout << "Reduce CPU: " << reduce_cpu << endl;
	TYPE reduce_gpu = ReduceGPU::reduce(arr, n, Operators::add<TYPE>());
	cout << "Reduce GPU: " << reduce_gpu << endl;

	TYPE* inc_scan_cpu = ScanCPU::inclusive_scan(arr, n, Operators::max<TYPE>());
	cout << "Inclusive Scan CPU: ";
	for (size_t i = 0; i < n; i += n >> 3)
		cout << inc_scan_cpu[i] << " ";
	cout << endl;
	TYPE* inc_scan_gpu = ScanGPUHillisSteele::inclusive_scan(arr, n, Operators::max<TYPE>());
	cout << "Inclusive Scan GPU: ";
	for (size_t i = 0; i < n; i += n >> 3)
		cout << inc_scan_gpu[i] << " ";
	cout << endl;

	delete [] inc_scan_cpu;
	delete [] inc_scan_gpu;

	delete [] arr;
}