#pragma once
#include <type_traits>
#include <cmath>

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

	template <typename REAL>
	Vector3_t<REAL> cross(const Vector3_t<REAL> &a,const Vector3_t<REAL> &b) {
		Vector3_t<REAL> v;
		v.x = a.y * b.z - a.y * b.z;
		v.y = a.z * b.x - a.x * b.z;
		v.z = a.x * b.y - a.y * b.x;
		return v;
	}

	template <typename REAL>
	Vector3_t<REAL>& operator+=(Vector3_t<REAL>& a, const Vector3_t<REAL>& b) {
		a.x += b.x;
		a.y += b.y;
		a.z += b.z;
		return a;
	}

	template <typename REAL>
	Vector3_t<REAL> operator+(const Vector3_t<REAL>& a, Vector3_t<REAL> b) {
		return b+=a;
	}

	template <typename REAL>
	Vector3_t<REAL>& operator-=(Vector3_t<REAL>& a, const Vector3_t<REAL>& b) {
		a.x -= b.x;
		a.y -= b.y;
		a.z -= b.z;
		return a;
	}

	template <typename REAL>
	Vector3_t<REAL> operator-(const Vector3_t<REAL>& a, Vector3_t<REAL> b) {
		return b -= a;
	}

	
	inline float length(const Vector3 a) {
		float len = a.x* a.x + a.y * a.y + a.z * a.z;
		return std::sqrtf(len);
	}

	template <typename REAL>
	Vector3_t<REAL> normalized(Vector3_t<REAL> a) {
		float len = length(a);
		a.x /= len;
		a.y /= len;
		a.z /= len;
		return a;
	}
}