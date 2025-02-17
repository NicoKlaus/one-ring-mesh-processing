#include <util.hpp>


using namespace std;
using namespace happly;
namespace ab {

	bool read_mesh(SimpleMesh& mesh, const string& file) {
		PLYData plyIn(file);
		plyIn.validate();
		std::vector<float> xPos = plyIn.getElement("vertex").getProperty<float>("x");
		std::vector<float> yPos = plyIn.getElement("vertex").getProperty<float>("y");
		std::vector<float> zPos = plyIn.getElement("vertex").getProperty<float>("z");

		auto& positions = mesh.positions;
		positions.resize(xPos.size());
		for (size_t i = 0; i < positions.size(); ++i) {
			positions[i].x = xPos[i];
			positions[i].y = yPos[i];
			positions[i].z = zPos[i];
		}
		std::vector<std::vector<int>> faces_vector = plyIn.getElement("face").getListProperty<int>("vertex_indices");

		//clear vectors
		mesh.face_starts.resize(0);
		mesh.faces.resize(0);
		mesh.face_sizes.resize(0);
		//encode face information inside 3 arrays
		for (auto face : faces_vector) {
			//store begin and size of the face
			mesh.face_sizes.emplace_back(face.size());
			mesh.face_starts.emplace_back(mesh.faces.size());
			//append face indices
			for (auto vert : face) {
				mesh.faces.emplace_back(vert);
			}
		}
		mesh.face_starts.push_back(mesh.faces.size());
		return true;
	}

	bool read_mesh(HalfedgeMesh& mesh, const std::string& file)
	{
		SimpleMesh s_mesh;
		if (!read_mesh(s_mesh, file)) return false;
		return create_he_mesh_from(mesh, s_mesh);
	}

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
		for (int i = 0; i < mesh.face_starts.size()-1; ++i) {
			std::vector<int> face;
			for (int j = 0; j < mesh.face_sizes[i]; ++j) {
				face.emplace_back(mesh.faces[mesh.face_starts[i] + j]);
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

	template <typename T>
	struct LessThanPairSecond {
		bool operator()(pair<T,T> a,pair<T,T> b) {
			return a.second < b.second;
		}
	};
	template <typename T>
	struct LessThanPairFirst {
		bool operator()(pair<T, T> a, pair<T, T> b) {
			return a.first < b.first;
		}
	};

	void sort_mesh(HalfedgeMesh &mesh) {
		attribute_vector<pair<int,int>> vertices_by_valence(mesh.vertices.size());
		for (int i = 0; i < mesh.vertices.size(); ++i) {
			vertices_by_valence[i].first = i;
			//find valenz
			int valenz = 0;
			int he = mesh.vertices[i].he;
			if (he >= 0) {
				do {
					he = mesh.half_edges[mesh.half_edges[he].inv].next;
					++valenz;
				} while (he != mesh.vertices[i].he);
			}
			vertices_by_valence[i].second = valenz;
		}
		std::sort(vertices_by_valence.begin(), vertices_by_valence.end(), LessThanPairSecond<int>());

		attribute_vector<pair<int, int>> permutation(vertices_by_valence.size());
		for (int i = 0; i < vertices_by_valence.size(); ++i) {
			permutation[i].first = vertices_by_valence[i].first; //old index
			permutation[i].second = i; //new index
		}

		//permutate vertices
		attribute_vector<Vertex> vertices = mesh.vertices;
		for (int i = 0; i < vertices_by_valence.size(); ++i) {
			mesh.vertices[i] = vertices[vertices_by_valence[i].first];
		}

		//fix halfedges
		sort(permutation.begin(), permutation.end(), LessThanPairFirst<int>());
		for (int i = 0; i < mesh.half_edges.size(); ++i) {
			mesh.half_edges[i].origin = permutation[mesh.half_edges[i].origin].second;
		}


		//cut orphaned vertices with valenz == 0
		int valid_valenz_index = 0;
		while (valid_valenz_index < vertices_by_valence.size()) {
			if (vertices_by_valence[valid_valenz_index].second != 0) {
				break;
			}
			valid_valenz_index++;
		}
		
		vertices = mesh.vertices;
		mesh.vertices.resize(vertices.size() - valid_valenz_index);
		copy(vertices.begin() + valid_valenz_index, vertices.end(), mesh.vertices.begin());
		for (int i = 0; i < mesh.half_edges.size(); ++i) {
			mesh.half_edges[i].origin-= valid_valenz_index;
		}
	}
}