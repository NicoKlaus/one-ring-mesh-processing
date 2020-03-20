#include <SimpleMesh.hpp>
#include <happly.h>
#include <cuda_mesh_operations.hpp>

using namespace std;
using namespace happly;


namespace ab {


	bool write_ply(const SimpleMesh &mesh, const std::string &file) {
		PLYData plyOut;
		plyOut.addElement("vertex",mesh.positions.size());
		
		//positions
		vector<float> coordsX;
		for (auto pos : mesh.positions) {
			coordsX.push_back(pos.x);
		}
		plyOut.getElement("vertex").addProperty<float>("x",coordsX);
	
		for (size_t i = 0; i < mesh.positions.size(); ++i) {
			coordsX[i] = mesh.positions[i].y;
		}
		plyOut.getElement("vertex").addProperty<float>("y", coordsX);
		
		for (size_t i = 0; i < mesh.positions.size(); ++i) {
			coordsX[i] = mesh.positions[i].z;
		}
		plyOut.getElement("vertex").addProperty<float>("z", coordsX);
		
		if (mesh.normals.size() > 0 && mesh.normals.size() == mesh.positions.size()) {
			coordsX.resize(mesh.normals.size());
			for (size_t i = 0; i < mesh.normals.size(); ++i) {
				coordsX[i] = mesh.normals[i].x;
			}
			plyOut.getElement("vertex").addProperty<float>("nx", coordsX);

			for (size_t i = 0; i < mesh.normals.size(); ++i) {
				coordsX[i] = mesh.normals[i].y;
			}
			plyOut.getElement("vertex").addProperty<float>("ny", coordsX);

			for (size_t i = 0; i < mesh.normals.size(); ++i) {
				coordsX[i] = mesh.normals[i].z;
			}
			plyOut.getElement("vertex").addProperty<float>("nz", coordsX);
		}

		//faces
		std::vector<std::vector<int>> faces_vector;
		for (int i = 0;i < mesh.face_starts.size();++i) {
			std::vector<int> face;
			for (int j = 0; j < mesh.face_sizes[i];++j) {
				face.emplace_back(mesh.faces[mesh.face_starts[i] + j]);
			}
			faces_vector.emplace_back(face);
		}
		plyOut.addElement("face", faces_vector.size());
		plyOut.getElement("face").addListProperty("vertex_indices", faces_vector);
		plyOut.write(file);
		return true;
	}

}