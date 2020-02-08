#include "mesh_normal_generator_vtkm.hpp"
#include <util.hpp>
#include <boost/program_options.hpp>
#include <SimpleMesh.hpp>
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
#include <vtkm\filter\SurfaceNormals.h>
#include <vtkm/cont/ArrayCopy.h>
using namespace std;
using namespace vtkm::cont;
using namespace vtkm;

namespace ab {

	void normals_by_area_sm_vtkm(SimpleMesh* mesh, timing_struct& timing) {
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
		ArrayHandle<UInt8> shapes = make_ArrayHandle(shapes_array.data(), shapes_array.size());

		ArrayHandle<Id> connectivity;
		vector<Id> connectivity_array;
		for (int i = 0; i < mesh->face_indices.size(); ++i) {
			connectivity_array.push_back(mesh->face_indices[i]);
		}
		connectivity = make_ArrayHandle(connectivity_array.data(), connectivity_array.size());

		ArrayHandle<IdComponent> numIndices = make_ArrayHandle(reinterpret_cast<IdComponent*>(mesh->face_sizes.data()), mesh->face_sizes.size());
		vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
		DataSet dataSet = vtkm::cont::DataSetBuilderExplicit::Create(positions, shapes, numIndices, connectivity);

		vtkm::filter::SurfaceNormals filter;
		filter.SetGeneratePointNormals(true);
		filter.SetPointNormalsName("PointNormals");
		auto start = std::chrono::steady_clock::now();
		DataSet result = filter.Execute(dataSet);
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		auto handle = result.GetField("PointNormals").GetData().Cast<ArrayHandle<Vec3f_32>>();
		ArrayCopy(handle, normals);
	}


	mesh_normal_generator_vtkm::mesh_normal_generator_vtkm(SimpleMesh* mesh)
	{
		mesh_pointer = mesh;
	}

	mesh_normal_generator_vtkm::~mesh_normal_generator_vtkm()
	{
	}

	void mesh_normal_generator_vtkm::operator()()
	{
		SimpleMesh* mesh = reinterpret_cast<SimpleMesh*>(mesh_pointer);
		normals_by_area_sm_vtkm(mesh, timings);
	}

}