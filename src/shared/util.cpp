#include <util.hpp>


using namespace std;
using namespace happly;
namespace ab {

	bool read_mesh(SimpleMesh& mesh, const string& file) {
		PLYData plyIn(file);
		plyIn.validate();
		mesh.positions_x = plyIn.getElement("vertex").getProperty<float>("x");
		mesh.positions_y = plyIn.getElement("vertex").getProperty<float>("y");
		mesh.positions_z = plyIn.getElement("vertex").getProperty<float>("z");

		std::vector<std::vector<int>> faces_vector = plyIn.getElement("face").getListProperty<int>("vertex_indices");

		//clear vectors
		mesh.faces.resize(0);
		mesh.face_indices.resize(0);
		mesh.face_sizes.resize(0);
		//encode face information inside 3 arrays
		for (auto face : faces_vector) {
			//store begin and size of the face
			mesh.face_sizes.emplace_back(face.size());
			mesh.faces.emplace_back(mesh.face_indices.size());
			//append face indices
			for (auto vert : face) {
				mesh.face_indices.emplace_back(vert);
			}
		}
		return true;
	}

	bool read_mesh(HalfedgeMesh& mesh, const std::string& file)
	{
		SimpleMesh s_mesh;
		if (!read_mesh(s_mesh, file)) return false;
		return create_he_mesh_from(mesh, s_mesh);
	}

	void write_pointcloud(const string& fn, float* points_x, float* points_y, float* points_z, size_t size,bool binary_mode)
	{
		PLYData plyOut;
		vector<float> x;
		x.resize(size);
		plyOut.addElement("vertex", size);
		memcpy(x.data(), points_x, size * sizeof(float));
		plyOut.getElement("vertex").addProperty<float>("x", x);
		memcpy(x.data(), points_y, size * sizeof(float));
		plyOut.getElement("vertex").addProperty<float>("y", x);
		memcpy(x.data(), points_z, size * sizeof(float));
		plyOut.getElement("vertex").addProperty<float>("z", x);
		plyOut.write(fn, binary_mode ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
	}


	bool write_mesh(const SimpleMesh& mesh, const std::string& file,bool binary_mode) {
		PLYData plyOut;
		plyOut.addElement("vertex", mesh.vertex_count());

		//positions
		plyOut.getElement("vertex").addProperty<float>("x", mesh.positions_x);
		plyOut.getElement("vertex").addProperty<float>("y", mesh.positions_y);
		plyOut.getElement("vertex").addProperty<float>("z", mesh.positions_z);

		if (mesh.normals_x.size() > 0 && mesh.normals_x.size() == mesh.vertex_count()) {
			plyOut.getElement("vertex").addProperty<float>("nx", mesh.normals_x);
			plyOut.getElement("vertex").addProperty<float>("ny", mesh.normals_y);
			plyOut.getElement("vertex").addProperty<float>("nz", mesh.normals_z);
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
	
	size_t in_memory_mesh_size(const SimpleMesh& mesh) {
		size_t size = 0;
		size += sizeof(int) * mesh.faces.size() +
			sizeof(int) * mesh.face_indices.size() +
			sizeof(int) * mesh.face_sizes.size() +
			3*sizeof(float) * mesh.normals_x.size() +
			3*sizeof(float) * mesh.positions_x.size();
		return size;
	}
	
	size_t in_memory_mesh_size(const HalfedgeMesh& mesh) {
		size_t size = 0;
		/*
		size += sizeof(HalfEdge) * mesh.half_edges.size() +
			sizeof(Vertex) * mesh.vertices.size() +
			sizeof(Loop) * mesh.loops.size() +
			sizeof(float) * mesh.normals.size();
		*/
		return size;
	}

}