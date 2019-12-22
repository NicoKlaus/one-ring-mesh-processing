#pragma once
#include <type_traits>

namespace ab {

	template <typename REAL>
	struct Vector3_t {
		REAL x, y, z;

		REAL& operator[](size_t i) {
			return reinterpret_cast<REAL*>(this)[i];
		}

		const REAL& operator[](size_t i) const {
			return reinterpret_cast<REAL*>(this)[i];
		}
	};

	typedef Vector3_t<float> Vector3;
}