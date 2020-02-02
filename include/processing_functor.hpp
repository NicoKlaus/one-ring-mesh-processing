#pragma once
#include <vector>
#include <timing_struct.hpp>

namespace ab {

	enum ProcessingDevice {
		PD_CPU = 1,
		PD_CUDA = 2,
		PD_NONE = 0
	};

	enum ProcessingMode {
		PM_SCATTER = 1,
		PM_GATHER = 2,
		PM_NONE = 0
	};

	class processing_functor {
	public:
		virtual ~processing_functor() = 0;
		virtual void operator()() = 0;

		size_t threads, blocks;
		timing_struct timings;
	protected:
		ProcessingMode proc_mode;
		ProcessingDevice proc_dev;
		void* mesh_pointer;
	};

}