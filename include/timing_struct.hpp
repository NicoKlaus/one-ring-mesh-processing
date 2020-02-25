#pragma once

namespace ab{

	struct timing_struct {
		size_t data_upload_time;
		///time needed for complete processing of the mesh, measured over the timespan of all kernel execution times + overheads from event creation and sync and time measureing
		size_t processing_time;
		size_t kernel_execution_time_a;//execution time needed by the first kernel
		size_t kernel_execution_time_b;//execution time needed by the second kernel if any
		size_t kernel_execution_time_prepare;//custom kernel for data preparation
		size_t sorting_time;
		size_t unique_time;
		size_t data_download_time;
		int block_size;
		int grid_size;

		inline timing_struct() : data_upload_time(0), kernel_execution_time_a(0), kernel_execution_time_b(0), kernel_execution_time_prepare(0),
			data_download_time(0), block_size(0),grid_size(0),processing_time(0),sorting_time(0),unique_time(0) {};
		inline size_t total_execution_time() {
			return data_upload_time + data_download_time + kernel_execution_time_a + kernel_execution_time_b;
		}
	};

}