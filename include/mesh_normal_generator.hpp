#pragma once
#include <processing_functor.hpp>

namespace ab {
	
	class SimpleMesh;
	class HalfedgeMesh;

	class mesh_normal_generator : public processing_functor {
	public:
		mesh_normal_generator(SimpleMesh* mesh, ProcessingDevice dev,size_t threads,size_t blocks=1);
		mesh_normal_generator(HalfedgeMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks=1);
		~mesh_normal_generator();
		void operator()();

		size_t threads, blocks;
	protected:
		ProcessingMode proc_mode;
		ProcessingDevice proc_dev;
		void* mesh_pointer;
	};

}