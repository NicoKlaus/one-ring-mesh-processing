#pragma once
#include <vector>

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
	protected:
		std::vector<size_t> timings;
	};

}