#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include "types.cuh"
#include "cuda_helpers.cuh"
#include "kernel_args.hpp"

namespace cg = cooperative_groups;

namespace cross {

namespace {

template<typename T>
__global__ void scatter(
    const T *__restrict__ src,
    T *__restrict__ dst,
    dsize2_t src_matrix_size,
    dsize_t src_num_matrices,
    dsize2_t dst_matrix_size,
    dsize2_t dst_pos
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block ctb = cg::this_thread_block();

    auto num_blocks = grid.group_dim().x;
    dsize_t total_area = src_matrix_size.area() * src_num_matrices;
    dsize_t area_per_block = (total_area / num_blocks) + 1;
    dsize_t block_start = area_per_block * ctb.group_index().x;
    dsize_t block_end = min(total_area, block_start + area_per_block);


    for (dsize_t i = block_start + ctb.thread_index().x; i < block_end; i += ctb.group_dim().x) {
        auto data = src[i];
        auto src_matrix = i / src_matrix_size.area();
        auto offset = i % src_matrix_size.area();
        auto src_y = offset / src_matrix_size.x;
        auto src_x = offset % src_matrix_size.y;

        auto dst_matrix_start = src_matrix * dst_matrix_size.area();
        auto dst_matrix_row_start = dst_matrix_start + (dst_pos.y + src_y) * dst_matrix_size.x;

        dst[dst_matrix_row_start + dst_pos.x + src_x] = data;
    }
}

/**
 * Args used for the kernel call. The class is a singleton to minimize the impact
 * on measured time (prevent allocation etc.)
 */
class scatter_kernel_args : public kernel_args {
public:
    scatter_kernel_args(const scatter_kernel_args&) = delete;
    scatter_kernel_args& operator=(scatter_kernel_args&) = delete;

    static void record_launch(
        dim3 block_size,
        dim3 grid_size
    ) {
        static scatter_kernel_args instance;
        instance.set_common(block_size, grid_size, 0);
        set_last_kernel_launch_args(&instance);
    }

private:
    scatter_kernel_args()
        : kernel_args()
    { }
};


} // END anonymous namespace

template<typename T>
void run_scatter(
    const T* __restrict__ src,
    T* __restrict__ dst,
    dsize2_t src_matrix_size,
    dsize_t src_num_matrices,
    dsize2_t dst_matrix_size,
    dsize2_t dst_pos,
    dsize_t threads_per_block,
    dsize_t items_per_threads
) {
    dsize_t total_area = src_matrix_size.area() * src_num_matrices;
    auto total_threads = total_area / items_per_threads;
    dsize_t num_blocks = div_up(total_threads, threads_per_block);
    scatter<<<num_blocks, threads_per_block>>>(
        src,
        dst,
        src_matrix_size,
        src_num_matrices,
        dst_matrix_size,
        dst_pos
    );

    scatter_kernel_args::record_launch(
        threads_per_block,
        num_blocks
    );
}


template void run_scatter<float>(
    const float* __restrict__ src,
    float* __restrict__ dst,
    dsize2_t src_matrix_size,
    dsize_t src_num_matrices,
    dsize2_t dst_matrix_size,
    dsize2_t dst_pos,
    dsize_t threads_per_block,
    dsize_t items_per_threads
);

// template void run_scatter<double>(
//     const double* __restrict__ src,
//     double* __restrict__ dst,
//     dsize2_t src_matrix_size,
//     dsize_t src_num_matrices,
//     dsize2_t dst_matrix_size,
//     dsize2_t dst_pos,
//     dsize_t threads_per_block,
//     dsize_t items_per_threads
// );

// template void run_scatter<int>(
//     const int* __restrict__ src,
//     int* __restrict__ dst,
//     dsize2_t src_matrix_size,
//     dsize_t src_num_matrices,
//     dsize2_t dst_matrix_size,
//     dsize2_t dst_pos,
//     dsize_t threads_per_block,
//     dsize_t items_per_threads
// );

}
