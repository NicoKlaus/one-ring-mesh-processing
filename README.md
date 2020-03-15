Options:
  -h [ --help ]          Help screen
  --in arg               Ply File for reading
  --out arg              Ply File for writing
  --test arg             runs tests,= Ply File for testing
  --algorithm arg        normals-gather-cuda|normals-scatter-cuda|centroids-gat
                         her-cuda|centroids-scatter-cuda
  --threads arg          threads per block, blocks and threads are determined 
                         automatically if ommited
  --blocks arg           blocks in the grid, has no effect for cpu only 
                         algorithms, determined automatically if --threads 
                         ommited
  --runs arg             =N ,run calculation N times for extensive time 
                         mesuring
  --time-log arg         saves timings to file
  --strip-attributes arg removes all attributes from the mesh except positions 
                         and connectivity
  --sort arg             sorts input mesh by veretx valenz



Build Boost

cd lib/<boost dir>
bootstrap
./b2

set CUDA arch

cmake .. -DCMAKE_CUDA_FLAGS="-arch=sm_30"