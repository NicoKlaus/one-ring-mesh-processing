#pragma once
#include <cuda_runtime.h>
#include <HalfEdgeMesh.hpp>
#include <SimpleMesh.hpp>
#include <cmath>
#include <vector>

namespace ab {
	void calculate_normals_he_parallel_area_weight(HalfedgeMesh* mesh, size_t threads = 1024);
	void calculate_normals_sm_parallel_area_weight(SimpleMesh* mesh, size_t threads = 1024);
	void calculate_centroids_he_parallel(HalfedgeMesh* mesh, std::vector<float3>& centroids_array, size_t threads = 1024);
	void calculate_centroids_sm_parallel(SimpleMesh* mesh, std::vector<float3>& centroids, size_t threads = 256, size_t blocks = 1);

	//float3 extensions
	__forceinline__ __host__ __device__ float3 cross3df(const float3 a, const float3 b) {
		float3 v;
		v.x = a.y * b.z - a.z * b.y;
		v.y = a.z * b.x - a.x * b.z;
		v.z = a.x * b.y - a.y * b.x;
		return v;
	}

	__forceinline__ __host__ __device__ float3& operator+=(float3& a, const float3& b) {
		a.x += b.x;
		a.y += b.y;
		a.z += b.z;
		return a;
	}

	__forceinline__ __host__ __device__ float3 operator+(const float3& a, float3 b) {
		return b += a;
	}
	
	__forceinline__ __host__ __device__ float3& operator-=(float3& a, const float3& b) {
		a.x -= b.x;
		a.y -= b.y;
		a.z -= b.z;
		return a;
	}

	__forceinline__ __host__ __device__ float3 operator-(const float3& a, float3 b) {
		return b -= a;
	}

	__forceinline__ __host__ __device__ float length(const float3& a) {
		#if defined(__CUDA_ARCH__)
			return norm3df(a.x, a.y, a.z);
		#else
			float len = a.x * a.x + a.y * a.y + a.z * a.z;
			return std::sqrtf(len);
		#endif
	}

	__forceinline__ __host__ __device__ float3 normalized(float3 a) {
		float len = length(a);
		a.x /= len;
		a.y /= len;
		a.z /= len;
		return a;
	}
}