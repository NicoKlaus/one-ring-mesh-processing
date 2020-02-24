#pragma once
#include <processing_functor.hpp>
#include <SimpleMesh.hpp>
#include <HalfedgeMesh.hpp>

namespace ab {

	class mesh_centroid_generator : public processing_functor {
	public:
		mesh_centroid_generator(SimpleMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks = 1);
		mesh_centroid_generator(HalfedgeMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks = 1);
		~mesh_centroid_generator();
		void operator()();
		attribute_vector<float3> centroids;
	};

}