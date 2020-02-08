#pragma once

namespace ab{

	struct timing_struct {
		size_t data_upload_time;
		size_t kernel_execution_time_a;
		size_t kernel_execution_time_b;
		size_t data_download_time;
		int block_size;
		int grid_size;

		inline timing_struct() : data_upload_time(0), kernel_execution_time_a(0), kernel_execution_time_b(0), data_download_time(0)
				, block_size(0),grid_size(0) {};
		inline size_t total_execution_time() {
			return data_upload_time + data_download_time + kernel_execution_time_a + kernel_execution_time_b;
		}
	};

}