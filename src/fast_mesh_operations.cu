#include <fast_mesh_operations.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

using namespace thrust;

namespace ab {
	__global__ void kernel_calculate_normals_no_weight(Vertex* vertices, HalfEdge* half_edges, Vector3* normals, unsigned vertice_count) {
		int stride = blockDim.x;
		int offset = threadIdx.x;/*
		//calculate normal without weight
		for (int i = 0; i < vertice_count; ++i) {
			auto& vert = vertices[i];
			if (vert.he == -1) {
				continue;
			}
			int he = vert.he;
			Vector3 normal{ 0.f,0.f,0.f };
			do {
				HalfEdge& halfedge = half_edges[he];
				Vector3 a = vertices[halfedge.origin].position;
				Vector3 b = vertices[half_edges[halfedge.next].origin].position;
				normal += cross(a, b);
				he = halfedge.next;
			} while (he != vert.he);
			normal = normalized(normal);
			normals[i] = normal;
		}*/
	}

	void calculate_normals_he_parallel_no_weight(HalfedgeMesh* mesh) {
		mesh->normals.resize(mesh->vertices.size());
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<Vector3> normals = mesh->normals;
		kernel_calculate_normals_no_weight<<<1,128>>>(vertices.data().get(), halfedges.data().get(), normals.data().get(), vertices.size());
		cudaDeviceSynchronize();
	}
}



