#ifndef Operators_cuh
#define Operators_cuh

#include <limits>

class Operators {
public:
	template<class T>
	class Add {
	public:
		__host__ __device__
		T operator()(T a, T b) const {
			return a + b;
		}

		const T identity;

		Add() : identity(static_cast<T>(0)) {}
	};

	template<class T>
	static Add<T> add() {
		return Add<T>();
	}

	template<class T>
	class Multiply {
	public:
		__host__ __device__
		T operator()(T a, T b) const {
			return a * b;
		}

		const T identity;

		Multiply() : identity(static_cast<T>(1)) {}
	};

	template<class T>
	static Multiply<T> multiply() {
		return Multiply<T>();
	}

	template<class T>
	class Max {
	public:
		__host__ __device__
		T operator()(T a, T b) const {
			return a > b ? a : b;
		}

		const T identity;

		Max() : identity(std::numeric_limits<T>::min()) {}
	};

	template<class T>
	static Max<T> max() {
		return Max<T>();
	}

	template<class T>
	class Min {
	public:
		__host__ __device__
		T operator()(T a, T b) const {
			return a < b ? a : b;
		}

		const T identity;

		Min() : identity(std::numeric_limits<T>::max()) {}
	};

	template<class T>
	static Min<T> min() {
		return Min<T>();
	}
};

#endif