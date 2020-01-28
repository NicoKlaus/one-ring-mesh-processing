#include <util.hpp>


using namespace std;
using namespace happly;
namespace ab {
	void write_pointcloud(const string& fn, float3* points, size_t size,bool binary_mode)
	{
		PLYData plyOut;
		plyOut.addElement("vertex", size);

		//positions
		vector<float> coordsX;
		for (size_t i = 0; i < size; ++i) {
			coordsX.push_back(points[i].x);
		}
		plyOut.getElement("vertex").addProperty<float>("x", coordsX);

		for (size_t i = 0; i < size; ++i) {
			coordsX[i] = points[i].y;
		}
		plyOut.getElement("vertex").addProperty<float>("y", coordsX);

		for (size_t i = 0; i < size; ++i) {
			coordsX[i] = points[i].z;
		}
		plyOut.getElement("vertex").addProperty<float>("z", coordsX);
		plyOut.write(fn, binary_mode ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
	}


	bool write_mesh(const SimpleMesh& mesh, const std::string& file,bool binary_mode) {
		PLYData plyOut;
		plyOut.addElement("vertex", mesh.positions.size());

		//positions
		vector<float> coordsX;
		for (auto pos : mesh.positions) {
			coordsX.push_back(pos.x);
		}
		plyOut.getElement("vertex").addProperty<float>("x", coordsX);

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
		for (int i = 0; i < mesh.faces.size(); ++i) {
			std::vector<int> face;
			for (int j = 0; j < mesh.face_sizes[i]; ++j) {
				face.emplace_back(mesh.face_indices[mesh.faces[i] + j]);
			}
			faces_vector.emplace_back(face);
		}
		plyOut.addElement("face", faces_vector.size());
		plyOut.getElement("face").addListProperty("vertex_indices", faces_vector);
		plyOut.write(file,binary_mode ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
		return true;
	}


	bool write_mesh(const HalfedgeMesh& mesh, const std::string& file, bool binary_mode)
	{
		SimpleMesh s_mesh;
		if (!create_simple_mesh_from(s_mesh, mesh)) return false;
		return write_mesh(s_mesh, file, binary_mode);
	}
}