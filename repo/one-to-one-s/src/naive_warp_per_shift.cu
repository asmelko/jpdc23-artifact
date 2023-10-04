#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdexcept>

#include "types.cuh"
#include "cuda_helpers.cuh"
#include "row_distribution.cuh"
#include "warp_size.hpp"
#include "kernel_args.hpp"


namespace cg = cooperative_groups;

namespace cross {

namespace {

template<typename T, typename RES>
__global__ void ccn_warp_per_shift(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_thread_block = ctb.group_dim().y;

    dsize2_t warp_out_pos{
        ctb.thread_index().y + ctb.group_index().x * shifts_per_thread_block,
        ctb.group_index().y
    };

    if (warp_out_pos.x >= search_size.x || warp_out_pos.y >= search_size.y) {
        return;
    }

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> warp_shift = {
        static_cast<int>(warp_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(warp_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    dsize2_t right_start(
        max(0, -warp_shift.x),
        max(0, -warp_shift.y)
    );

    dsize2_t right_end(
        min(matrix_size.x - warp_shift.x, matrix_size.x),
        min(matrix_size.y - warp_shift.y, matrix_size.y)
    );

    dsize2_t overlap_size = right_end - right_start;
    dsize_t total_items = overlap_size.area();

    RES sum = 0;
    // Simpler internal loop, as is done in simple_indexing version,
    // leads to high thread divergence and much slower overall speed
    // so even though this is bottlenecked by the index computations,
    // it still runs much faster
    for (dsize_t i = warp.thread_rank(); i < total_items; i += warp.size()) {
        dsize_t overlap_row = i / overlap_size.x;
        dsize_t overlap_row_offset = i % overlap_size.x;

        dsize2_t right_idx = right_start + dsize2_t{overlap_row_offset, overlap_row};
        dsize2_t left_idx = dsize2_t{
            right_idx.x + warp_shift.x,
            right_idx.y + warp_shift.y
        };


        sum += left[left_idx.linear_idx(matrix_size.x)] * right[right_idx.linear_idx(matrix_size.x)];
    }

    sum = cg::reduce(warp, sum, cg::plus<RES>());
    if (warp.thread_rank() == 0) {
        out[warp_out_pos.linear_idx(search_size.x)] = sum;
    }
}

/**
 *
 * @tparam T
 * @tparam RES
 * @param left
 * @param right
 * @param out Buffer for the output matrix, MUST BE ZEROED out
 * @param matrix_size
 * @param search_size
 * @param max_rows_per_warp
 */
template<typename DIST, typename T, typename RES>
__global__ void ccn_warp_per_shift_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t max_rows_per_warp
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_thread_block = ctb.group_dim().y;

    // Distribute rows of a single shift between multiple workers,
    // in this case warps
    // Return the assigned output row (which corresponds to a shift),
    // together with the number of workers computing this shift and
    // index of the current worker in range [0, number_of_workers_for_shift)
    assigned_work work = DIST::distribute_rows(
        ctb.group_index().y,
        max_rows_per_warp,
        matrix_size.y,
        search_size.y
    );

    dsize2_t warp_out_pos{
        ctb.thread_index().y + ctb.group_index().x * shifts_per_thread_block,
        work.output_row
    };

    // Either explicit check here for workers with idx above workers_for_row, or clamp right_start to be at most shared_right_end
    if (work.worker_idx >= work.workers_for_row || warp_out_pos.x >= search_size.x || warp_out_pos.y >= search_size.y) {
        return;
    }

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> warp_shift = {
        static_cast<int>(warp_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(warp_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    // Properties shared by all workers computing this shift
    dsize2_t shared_right_start(
        max(0, -warp_shift.x),
        max(0, -warp_shift.y)
    );

    dsize2_t shared_right_end(
        min(matrix_size.x - warp_shift.x, matrix_size.x),
        min(matrix_size.y - warp_shift.y, matrix_size.y)
    );

    dsize_t shared_overlapping_rows = shared_right_end.y - shared_right_start.y;
    dsize_t rows_per_worker = div_up(shared_overlapping_rows, work.workers_for_row);

    // Properties specific for the current worker
    dsize2_t right_start(
        shared_right_start.x,
        shared_right_start.y + work.worker_idx * rows_per_worker
    );

    dsize2_t right_end(
        shared_right_end.x,
        min(right_start.y + rows_per_worker, shared_right_end.y)
    );

    dsize2_t overlap_size = right_end - right_start;
    dsize_t total_items = overlap_size.area();

    RES sum = 0;
    for (dsize_t i = warp.thread_rank(); i < total_items; i += warp.size()) {
        dsize_t overlap_row = i / overlap_size.x;
        dsize_t overlap_row_offset = i % overlap_size.x;

        dsize2_t right_idx = right_start + dsize2_t{overlap_row_offset, overlap_row};
        dsize2_t left_idx = dsize2_t{
            right_idx.x + warp_shift.x,
            right_idx.y + warp_shift.y
        };


        sum += left[left_idx.linear_idx(matrix_size.x)] * right[right_idx.linear_idx(matrix_size.x)];
    }

    sum = cg::reduce(warp, sum, cg::plus<RES>());
    if (warp.thread_rank() == 0) {
        atomicAdd(out + warp_out_pos.linear_idx(search_size.x), sum);
    }
}

template<typename T, typename RES>
__global__ void ccn_warp_per_shift_simple_indexing(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_thread_block = ctb.group_dim().y;

    dsize2_t warp_out_pos{
        ctb.thread_index().y + ctb.group_index().x * shifts_per_thread_block,
        ctb.group_index().y
    };

    if (warp_out_pos.x >= search_size.x || warp_out_pos.y >= search_size.y) {
        return;
    }

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> warp_shift = {
        static_cast<int>(warp_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(warp_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    dsize2_t right_start(
        max(0, -warp_shift.x),
        max(0, -warp_shift.y)
    );

    dsize2_t right_end(
        min(matrix_size.x - warp_shift.x, matrix_size.x),
        min(matrix_size.y - warp_shift.y, matrix_size.y)
    );

    RES sum = 0;
    for (dsize_t right_y = right_start.y; right_y < right_end.y; ++right_y) {
        for (dsize_t right_x = right_start.x + warp.thread_rank(); right_x < right_end.x; right_x += warp.size()) {
            auto left_x = right_x + warp_shift.x;
            auto left_y = right_y + warp_shift.y;

            sum += left[left_y * matrix_size.x + left_x] * right[right_y * matrix_size.x + right_x];
        }
    }

    sum = cg::reduce(warp, sum, cg::plus<RES>());
    if (warp.thread_rank() == 0) {
        out[warp_out_pos.linear_idx(search_size.x)] = sum;
    }
}

/**
 * Args used for the kernel call. The class is a singleton to minimize the impact
 * on measured time (prevent allocation etc.)
 */
class warp_per_shift_kernel_args : public kernel_args {
public:
    warp_per_shift_kernel_args(const warp_per_shift_kernel_args&) = delete;
    warp_per_shift_kernel_args& operator=(warp_per_shift_kernel_args&) = delete;

    static void record_launch(
        dim3 block_size,
        dim3 grid_size
    ) {
        static warp_per_shift_kernel_args instance;
        instance.set_common(block_size, grid_size, 0);
        set_last_kernel_launch_args(&instance);
    }

private:
    warp_per_shift_kernel_args()
        : kernel_args()
    { }
};

/**
 * Args used for the kernel call. The class is a singleton to minimize the impact
 * on measured time (prevent allocation etc.)
 */
class ccn_warp_per_shift_work_distribution_kernel_args : public kernel_args {
public:
    distribution dist_;

    ccn_warp_per_shift_work_distribution_kernel_args(const ccn_warp_per_shift_work_distribution_kernel_args&) = delete;
    ccn_warp_per_shift_work_distribution_kernel_args& operator=(ccn_warp_per_shift_work_distribution_kernel_args&) = delete;

    static void record_launch(
        dim3 block_size,
        dim3 grid_size,
        distribution dist
    ) {
        static ccn_warp_per_shift_work_distribution_kernel_args instance;
        instance.set_common(block_size, grid_size, 0);
        instance.dist_ = dist;
        set_last_kernel_launch_args(&instance);
    }

    [[nodiscard]] std::unordered_map<std::string, std::string> get_additional_args() const override {
        return std::unordered_map<std::string, std::string>{
            {"work_distribution", to_string(dist_)}
        };
    }

private:
    ccn_warp_per_shift_work_distribution_kernel_args()
        : kernel_args(),
        dist_(distribution::none)
    { }
};

} // END anonymous namespace

template<typename T, typename RES>
void run_ccn_warp_per_shift(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block
) {
    if (shifts_per_thread_block > 32) {
        throw std::runtime_error("Too many shifts per thread block: "s + std::to_string(shifts_per_thread_block) + " (max 32)");
    }

    dim3 num_threads(warp_size, shifts_per_thread_block);
    dim3 num_blocks(
        div_up(search_size.x, num_threads.y),
        search_size.y
    );

    ccn_warp_per_shift<<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size
    );

    warp_per_shift_kernel_args::record_launch(num_threads, num_blocks);
}

template<typename DIST, typename T, typename RES>
void run_ccn_warp_per_shift_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block,
    dsize_t max_rows_per_warp
) {
    if (shifts_per_thread_block > 32) {
        throw std::runtime_error("Too many shifts per thread block: "s + std::to_string(shifts_per_thread_block) + " (max 32)");
    }

    dsize_t num_workers = DIST::num_workers(max_rows_per_warp, matrix_size.y, search_size.y);

    dim3 num_threads(warp_size, shifts_per_thread_block);
    dim3 num_blocks(
        div_up(search_size.x, num_threads.y),
        num_workers
    );

    ccn_warp_per_shift_work_distribution<DIST><<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        max_rows_per_warp
    );

    ccn_warp_per_shift_work_distribution_kernel_args::record_launch(num_threads, num_blocks, DIST::type);
}

template<typename T, typename RES>
void run_ccn_warp_per_shift_simple_indexing(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block
) {
    if (shifts_per_thread_block > 32) {
        throw std::runtime_error("Too many shifts per thread block: "s + std::to_string(shifts_per_thread_block) + " (max 32)");
    }

    dim3 num_threads(warp_size, shifts_per_thread_block);
    dim3 num_blocks(
        div_up(search_size.x, num_threads.y),
        search_size.y
    );

    ccn_warp_per_shift_simple_indexing<<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size
    );

    warp_per_shift_kernel_args::record_launch(num_threads, num_blocks);
}

// template void run_ccn_warp_per_shift<int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block
// );

template void run_ccn_warp_per_shift<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block
);

// template void run_ccn_warp_per_shift<double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block
// );

// template void run_ccn_warp_per_shift_simple_indexing<int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block
// );

template void run_ccn_warp_per_shift_simple_indexing<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block
);

// template void run_ccn_warp_per_shift_simple_indexing<double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block
// );

// template void run_ccn_warp_per_shift_work_distribution<triangle_distribution, int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block,
//     dsize_t max_rows_per_warp
// );

template void run_ccn_warp_per_shift_work_distribution<triangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block,
    dsize_t max_rows_per_warp
);

// template void run_ccn_warp_per_shift_work_distribution<triangle_distribution, double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block,
//     dsize_t max_rows_per_warp
// );

// template void run_ccn_warp_per_shift_work_distribution<rectangle_distribution, int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block,
//     dsize_t max_rows_per_warp
// );

template void run_ccn_warp_per_shift_work_distribution<rectangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block,
    dsize_t max_rows_per_warp
);

// template void run_ccn_warp_per_shift_work_distribution<rectangle_distribution, double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block,
//     dsize_t max_rows_per_warp
// );

// template void run_ccn_warp_per_shift_work_distribution<no_distribution, int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block,
//     dsize_t max_rows_per_warp
// );

template void run_ccn_warp_per_shift_work_distribution<no_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block,
    dsize_t max_rows_per_warp
);

// template void run_ccn_warp_per_shift_work_distribution<no_distribution, double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t shifts_per_thread_block,
//     dsize_t max_rows_per_warp
// );


}
