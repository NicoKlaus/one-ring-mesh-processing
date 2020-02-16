#pragma once
#include <cuda_runtime.h>
#include <HalfEdgeMesh.hpp>
#include <SimpleMesh.hpp>
#include <cmath>
#include <vector>
#include <timing_struct.hpp>

namespace ab {
	void normals_by_area_weight_he_cuda(HalfedgeMesh* mesh, int threads = 256,int blocks=1, timing_struct& timing=timing_struct());
	void normals_by_area_weight_he_cuda_no_stride(HalfedgeMesh* mesh, timing_struct& timing=timing_struct());
	void normals_by_area_weight_sm_cuda(SimpleMesh* mesh, int threads = 256, int blocks = 1, timing_struct& timing = timing_struct());
	void centroids_he_cuda(HalfedgeMesh* mesh, 
		std::vector<float>& centroids_array_x, std::vector<float>& centroids_array_y, std::vector<float>& centroids_array_z,
		int threads = 256,int blocks = 1, timing_struct& timing = timing_struct());
	void centroids_sm_cuda(SimpleMesh* mesh,
		std::vector<float>& centroids_array_x, std::vector<float>& centroids_array_y, std::vector<float>& centroids_array_z,
		int threads = 256, int blocks = 1, timing_struct& timing = timing_struct());

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