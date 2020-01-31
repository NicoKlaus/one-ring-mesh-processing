#pragma once

namespace ab{

	struct timing_struct {
		size_t data_upload_time;
		size_t kernel_execution_time_a;
		size_t kernel_execution_time_b;
		size_t data_download_time;
	};

}