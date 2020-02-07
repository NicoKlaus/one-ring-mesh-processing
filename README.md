Build Boost

cd lib/<boost dir>
bootstrap
./b2

set CUDA arch

cmake .. -DCMAKE_CUDA_FLAGS="-arch=sm_30"