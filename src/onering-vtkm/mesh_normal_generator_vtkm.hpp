#pragma once
#include <processing_functor.hpp>

namespace ab {
	
	class SimpleMesh;

	class mesh_normal_generator_vtkm : public processing_functor {
	public:
		mesh_normal_generator_vtkm(SimpleMesh* mesh);
		~mesh_normal_generator_vtkm();
		void operator()();
	};

}