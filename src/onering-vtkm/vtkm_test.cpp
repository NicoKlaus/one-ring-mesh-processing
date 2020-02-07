#include <util.hpp>
#include <boost/program_options.hpp>
#include <SimpleMesh.hpp>
#include <HalfEdgeMesh.hpp>
#include <mesh_centroid_generator.hpp>
#include <mesh_normal_generator.hpp>
#include <cuda_mesh_operations.hpp>
#include <cpu_mesh_operations.hpp>
#include <iostream>
#include <string>

#include <vtkm\cont\Initialize.h>
#include <vtkm\Types.h>
#include <vtkm\cont\ArrayHandle.h>
#include <vtkm\cont\DataSetBuilderExplicit.h>

using namespace std;
using namespace vtkm::cont;
using namespace vtkm;

namespace ab{
	
	int main(int argc,char* argv[]){
		vtkm::cont::Initialize();
		return 0;
	}
	
	void normals_by_area_sm_vtkm(SimpleMesh* mesh, int threads, timing_struct& timing) {
		mesh->normals.clear();
		mesh->normals.resize(mesh->positions.size(), float3{ 0.f,0.f,0.f });
		//Preparing the data set
		ArrayHandle<Vec3f_32> normals = 
				make_ArrayHandle(reinterpret_cast<Vec3f_32*>(mesh->normals.data()), mesh->normals.size());
		ArrayHandle<Vec3f_32> positions =
			make_ArrayHandle(reinterpret_cast<Vec3f_32*>(mesh->positions.data()), mesh->positions.size());
		vector<vtkm::UInt8 > shapes_array;
		for (auto face : mesh->faces) {
			shapes_array.push_back(vtkm::CELL_SHAPE_POLYGON);
		}
		ArrayHandle<UInt8> shapes = make_ArrayHandle(shapes_array.data(),shapes_array.size());
		
		ArrayHandle<Id> connectivity;
		vector<Id> connectivity_array;
		for (int i = 0; i < mesh->face_indices.size(); ++i) {
			connectivity_array.push_back(mesh->face_indices[i]);
		}
		connectivity = make_ArrayHandle(connectivity_array.data(), connectivity_array.size());
		
		ArrayHandle<IdComponent> numIndices = make_ArrayHandle(reinterpret_cast<IdComponent*>(mesh->face_sizes.data()), mesh->face_sizes.size());
		vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
		DataSet dataSet  = vtkm::cont::DataSetBuilderExplicit::Create(positions, shapes, numIndices, connectivity);
	}
}