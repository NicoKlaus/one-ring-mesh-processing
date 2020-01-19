#pragma once
#include <chrono>

namespace ab{
namespace perf{

template <typename F>
std::chrono::nanoseconds execution_time(F& lambda){
	auto start = std::chrono::high_resolution_clock::now();
	lambda();
	auto end = std::chrono::high_resolution_clock::now();
	return  std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
}

}
}