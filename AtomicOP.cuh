#ifndef AtomicOP_cuh
#define AtomicOP_cuh

template<class T, class Operator, size_t size>
struct AtomicOP {};

template<class T, class Operator>
struct AtomicOP<T, Operator, 4> {
	__device__ static T operate(T* address, T val, Operator op) {
		unsigned int* address_as_uint = (unsigned int*)address;
		unsigned int old = *address_as_uint, assumed;
		T current;
		do {
			assumed = old;
			current = op(val, *((T*)&assumed));
			old = atomicCAS(address_as_uint, assumed, *((unsigned int*)&current));
		} while (assumed != old);
		return *((T*)&old);
	}
};

template<class T, class Operator>
struct AtomicOP<T, Operator, 8> {
	__device__ static T operate(T* address, T val, Operator op) {
		unsigned long long int* address_as_uint = (unsigned long long int*)address;
		unsigned long long int old = *address_as_uint, assumed;
		T current;
		do {
			assumed = old;
			current = op(val, *((T*)&assumed));
			old = atomicCAS(address_as_uint, assumed, *((unsigned long long int*)&current));
		} while (assumed != old);
		return *((T*)&old);
	}
};

template<class T, class Operator>
__device__ T atomicOP(T* address, T val, Operator op) {
	return AtomicOP<T, Operator, sizeof(T)>::operate(address, val, op);
}

#endif