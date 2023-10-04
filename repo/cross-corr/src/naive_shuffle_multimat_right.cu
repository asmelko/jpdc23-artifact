#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <stdexcept>
#include <cassert>

#include "types.cuh"
#include "cuda_helpers.cuh"
#include "bound_checked_loads.cuh"

#include "row_distribution.cuh"
#include "warp_size.hpp"
#include "kernel_args.hpp"

namespace cg = cooperative_groups;

namespace cross {

namespace {

constexpr dsize_t right_matrices_per_thread_limit = SHUFFLE_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT;

__device__ void get_matrix_group(
    dsize_t output_size,
    dsize_t ctb_index,
    dsize_t matrices_per_thread,
    dsize_t& warp_output_offset,
    dsize_t& matrix_group_start_idx
) {
    // Matrix group is the group of right matrices (of at most right_matrices_per_thread matrices)
    // for which current thread computes the given shift
    // All warps in a block process the same 32 shifts in the x axis, but on different rows
    // so warps in the first block compute shifts 0-31, warps in the second block compute shifts 32-63 etc.
    // So each matrix_group needs to have as many threads as there are shifts in the x axis
    // so number of shifts in the x axis / warp_size
    // TODO: This is precomputed on CPU so we could pass it from there
    dsize_t blocks_per_matrix_group = div_up(output_size, warp_size);

    // Which matrix group this block and all its warps will compute
    dsize_t matrix_group_idx = ctb_index / blocks_per_matrix_group;
    // Offset of the current block and all of its warps in its matrix group
    // This corresponds to the position to write to in the output and the shift
    // to compute
    dsize_t matrix_group_block_offset = ctb_index % blocks_per_matrix_group;
    warp_output_offset = matrix_group_block_offset * warp_size;

    // Index of the first matrix in the group processed by the current thread
    matrix_group_start_idx = matrix_group_idx * matrices_per_thread;
}

/**
 * Arguments for the warp_shuffle_impl function.
 * As we need to write many calls for different constant values of NUM_RIGHTS which
 * all share the same argument values, we want to have each call as short as possible
 * This way, we can create the arguments with a single call and then use it in any of the calls in the switch statement
 *
 * @tparam T
 * @tparam RES
 */
template<typename T, typename RES>
struct warp_shuffle_impl_args {
    const T* __restrict__ left;
    const T* __restrict__ right;
    RES* __restrict__ out;
    dsize2_t warp_right_start;
    dsize2_t warp_right_end;
    vec2<int> warp_min_shift;
    dsize2_t output_pos;
    dsize2_t matrix_size;
    dsize2_t search_size;

    __device__ warp_shuffle_impl_args(
        const T* __restrict__ left,
        const T* __restrict__ right,
        RES* __restrict__ out,
        dsize2_t warp_right_start,
        dsize2_t warp_right_end,
        vec2<int> warp_min_shift,
        dsize2_t output_pos,
        dsize2_t matrix_size,
        dsize2_t search_size
    ) : left(left), right(right), out(out), warp_right_start(warp_right_start),
        warp_right_end(warp_right_end), warp_min_shift(warp_min_shift), output_pos(output_pos),
        matrix_size(matrix_size), search_size(search_size) {

    }
};

template<typename T, typename RES>
__device__ warp_shuffle_impl_args<T, RES> create_warp_shuffle_impl_args(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t warp_right_start,
    dsize2_t warp_right_end,
    vec2<int> warp_min_shift,
    dsize2_t output_pos,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    return warp_shuffle_impl_args<T, RES>(
        left,
        right,
        out,
        warp_right_start,
        warp_right_end,
        warp_min_shift,
        output_pos,
        matrix_size,
        search_size
    );
}

template<dsize_t NUM_RIGHTS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void warp_shuffle_impl(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args
) {
    // Compute the given shift for num_rights right matrices
    RES sum[NUM_RIGHTS];
    for (dsize_t i = 0; i < NUM_RIGHTS; ++i) {
        sum[i] = 0;
    }

    for (dsize_t warp_y_right = args.warp_right_start.y; warp_y_right < args.warp_right_end.y; warp_y_right += 1) {
        // In y axis, both max and min shift are equal in the current implementation
        dsize_t warp_y_left = warp_y_right + args.warp_min_shift.y;

        const dsize_t right_row_offset = warp_y_right * args.matrix_size.x;
        const T* left_row = args.left + warp_y_left * args.matrix_size.x;

        int warp_x_left = static_cast<int>(args.warp_right_start.x) + args.warp_min_shift.x;

        // Preload the first values from left matrix
        T thread_left_bottom = load_with_bounds_check(
            left_row,
            warp_x_left + warp.thread_rank(),
            args.matrix_size.x
        );

        for (
            dsize_t warp_x_right = args.warp_right_start.x;
            warp_x_right < args.warp_right_end.x;
            warp_x_right += warp.size(), warp_x_left += warp.size()
        ) {

            // Load next warp_size values
            // Load 0 if out of bounds

            // Right index will always be greater than 0 as we only
            // iterate over part of the matrix
            dsize_t right_idx = warp_x_right + warp.thread_rank();

            // Left index might be out of bounds even below 0, depending on the shift
            // It is also reading warp.size() next values, as we have warp.size() values already loaded
            // from the initialization before the for loop
            int left_idx = warp_x_left + warp.thread_rank() + warp.size();

            // Load values from num_rights right matrices
            T thread_right[NUM_RIGHTS];
            for (dsize_t r = 0; r < NUM_RIGHTS; ++r) {
                const T* matrix_start = args.right + r * args.matrix_size.area();
                const T* row = matrix_start + right_row_offset;
                // TODO: Either do bounds check or limit the for loop below
                thread_right[r] = load_with_bounds_check(row, right_idx, args.matrix_size.x);
            }

            T thread_left_top = load_with_bounds_check(left_row, left_idx, args.matrix_size.x);

            for (dsize_t i = 0; i < warp.size(); ++i) {
                for (dsize_t r = 0; r < NUM_RIGHTS; ++r) {
                    // Broadcast
                    auto right_val = warp.shfl(thread_right[r], i);

                    // No need to mask, if either values is out of bounds the value will be 0
                    sum[r] += thread_left_bottom * right_val;
                }

                // Shuffle does modulo srcLane automatically
                // Lane 0 pushes the bottom-most value of the top buffer to the top of the bottom buffer
                //  making it behave as one continuous buffer
                thread_left_bottom = warp.shfl(
                    warp.thread_rank() != 0 ? thread_left_bottom : thread_left_top,
                    warp.thread_rank() + 1
                );
                thread_left_top = warp.shfl_down(thread_left_top, 1);
            }
        }
    }

    if (args.output_pos.x < args.search_size.x && args.output_pos.y < args.search_size.y) {
        auto output_offset = args.output_pos.linear_idx(args.search_size.x);
        for (dsize_t r = 0; r < NUM_RIGHTS; ++r) {
            T* matrix = args.out + r * args.search_size.area();
            if (ATOMIC) {
                atomicAdd(matrix + output_offset, sum[r]);
            } else {
                matrix[output_offset] = sum[r];
            }
        }
    }
}

template<dsize_t NUM_RIGHTS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void warp_shuffle_impl_right_mats_dispatch(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t num_right_matrices,
    warp_shuffle_impl_args<T, RES> args
) {
    if constexpr(NUM_RIGHTS == 0) {
        (void)warp;
        (void)num_right_matrices;
        (void)args;
        assert(false);
    } else {
        if (NUM_RIGHTS == num_right_matrices) {
            warp_shuffle_impl<NUM_RIGHTS, ATOMIC>(warp, args);
        } else {
            warp_shuffle_impl_right_mats_dispatch<NUM_RIGHTS - 1, ATOMIC>(
                warp, num_right_matrices, args
            );
        }
    }
}

/**
 * This kernel first computes the range which should be
 * computed by the current warp in the left and right matrices
 * and then always loads 32 values
 */
template<dsize_t MAX_RIGHT_MATRICES_PER_THREAD, typename T, typename RES>
__global__ void ccn_shuffle_multimat_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices
) {
    // Initialize by loading a warp worth of data from left matrix
    // as we will be iterating over the left matrix

    // Then broadcast from the right data in sequence from all threads
    // With each broadcast, multiply and sum with the current value from
    // left matrix and then shuffle down the used values from left matrix.
    // Then shuffle the second warp worth of data from left matrix,
    // passing the last thread the value that is shuffled out of the thread 0
    // and would be forgotten
    // basically with warp size 4, it will go
    // 0 1 2 3 0 1 2 3, then 1 2 3 0 1 2 3 x, then 2 3 0 1 2 3 x x,
    // each time broadcasting first from thread 0, then 1, then 2
    // Once we get to 0 1 2 3 x x x x, we load one warp worth of values
    // from both left and right matrices

    // If the shift computed by the current thread does not overlap with the broadcast value
    // that means it tries to read from the left matrix out of bounds and thus will read 0
    // and ignore the broadcast value
    // By shifting the values down, when it reaches the part that overlaps it will receive
    // value shifted from the previous thread

    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);


    // Offset in the given output matrix on the x axis
    dsize_t output_x_offset;
    // Index of the first matrix in the group processed by the current thread
    dsize_t matrix_group_start_idx;
    get_matrix_group(
        search_size.x,
        ctb.group_index().x,
        MAX_RIGHT_MATRICES_PER_THREAD,
        output_x_offset,
        matrix_group_start_idx
    );


    // All warps of given block start at the same x, but each work on different row of output
    dsize2_t thread0_out_pos{
        output_x_offset,
        ctb.group_index().y * ctb.group_dim().y + ctb.thread_index().y
    };
    dsize2_t last_warp_thread_out_pos = thread0_out_pos +
                                        dsize2_t{warp.size() - 1, 0};

    // Position in the output matrix
    // This is unique for each thread, as each thread computes a single shift which
    // corresponds to a single output value
    dsize2_t output_pos = thread0_out_pos +
                          dsize2_t{warp.thread_rank(), 0};

    dsize2_t half_search_size = (search_size - 1) / 2;

    // Min of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 0
    vec2<int> warp_min_shift = {
        static_cast<int>(thread0_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(thread0_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    // Max of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 31
    // It is clamped into search size as matrix may not be of size divisible by warp_size
    vec2<int> warp_max_shift = {
        static_cast<int>(min(last_warp_thread_out_pos.x, search_size.x)) - static_cast<int>(half_search_size.x),
        static_cast<int>(min(last_warp_thread_out_pos.y, search_size.y)) - static_cast<int>(half_search_size.y)
    };


    // The start depends on the how far right the right matrix is shifted over the left matrix
    // if the right most shift, aka max shift is positive, that means that the left side of the right
    // matrix is inside the left matrix, so we need to start from the 0 element
    // if the max shift is negative, then absolute value tells us how many items of the right matrix are not needed
    // as they do not overlap in any shift computed by the matrix, as all smaller shifts have the right matrix more to the left
    // so they overlap less values
    dsize_t warp_x_right_start = warp_max_shift.x >= 0 ? 0 : -warp_max_shift.x;

    // The last value will be read by the min shift, so if it is larger than 0, the right side of the right matrix overhangs
    // the left matrix and so we don't need to reed the last abs(min_shift) values. Otherwise the right side of the right
    // matrix is inside the left matrix and we need to read it till the end.
    dsize_t warp_x_right_end = warp_min_shift.x >= 0 ? matrix_size.x - warp_min_shift.x : matrix_size.x;

    // All threads in a warp process the same range of rows, so warp_min_shift.y and warp_max_shift.y are the same
    dsize_t warp_y_right_start = max(-warp_min_shift.y, 0);
    dsize_t warp_y_right_end = min(matrix_size.y - warp_max_shift.y, matrix_size.y);

    dsize_t thread_num_right_matrices = min(num_right_matrices - matrix_group_start_idx, MAX_RIGHT_MATRICES_PER_THREAD);

    auto args = create_warp_shuffle_impl_args(
        left,
        right + matrix_group_start_idx * matrix_size.area(),
        out + matrix_group_start_idx * search_size.area(),
        dsize2_t{warp_x_right_start, warp_y_right_start},
        dsize2_t{warp_x_right_end, warp_y_right_end},
        warp_min_shift,
        output_pos,
        matrix_size,
        search_size
    );

    warp_shuffle_impl_right_mats_dispatch<MAX_RIGHT_MATRICES_PER_THREAD, false>(
        warp,
        thread_num_right_matrices,
        args
    );
}

/**
 * Args used for the kernel call. The class is a singleton to minimize the impact
 * on measured time (prevent allocation etc.)
 */
class ccn_shuffle_multimat_right_kernel_args : public kernel_args {
public:
    dsize_t max_right_matrices_per_thread_;

    ccn_shuffle_multimat_right_kernel_args(const ccn_shuffle_multimat_right_kernel_args&) = delete;
    ccn_shuffle_multimat_right_kernel_args& operator=(ccn_shuffle_multimat_right_kernel_args&) = delete;

    static void record_launch(
        dim3 block_size,
        dim3 grid_size,
        dsize_t max_right_matrices_per_thread
    ) {
        static ccn_shuffle_multimat_right_kernel_args instance;
        instance.set_common(block_size, grid_size, 0);
        instance.max_right_matrices_per_thread_ = max_right_matrices_per_thread;

        set_last_kernel_launch_args(&instance);
    }

    [[nodiscard]] std::unordered_map<std::string, std::string> get_additional_args() const override {
        return std::unordered_map<std::string, std::string>{
            {"max_right_matrices_per_thread", std::to_string(max_right_matrices_per_thread_)}
        };
    }

private:
    ccn_shuffle_multimat_right_kernel_args()
        : kernel_args(),
          max_right_matrices_per_thread_(0)
    { }
};


template<dsize_t MAX_RIGHT_MATRICES_PER_THREAD, typename T, typename RES>
__host__ void ccn_shuffle_multimat_right_right_mats_dispatch(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread
) {
    if constexpr(MAX_RIGHT_MATRICES_PER_THREAD == 0) {
        // Silence the unused parameter warning
        (void)left;
        (void)right;
        (void)out;
        (void)matrix_size;
        (void)search_size;
        (void)num_right_matrices;
        (void)warps_per_thread_block;
        (void)right_matrices_per_thread;
        assert(false);
    } else {
        if (MAX_RIGHT_MATRICES_PER_THREAD == right_matrices_per_thread) {
            dim3 num_threads(warp_size, warps_per_thread_block);

            dsize_t num_matrix_groups = div_up(num_right_matrices, right_matrices_per_thread);
            dsize_t blocks_per_matrix_group = div_up(search_size.x, num_threads.x);


            dim3 num_blocks(
                blocks_per_matrix_group * num_matrix_groups,
                div_up(search_size.y, num_threads.y)
            );

            ccn_shuffle_multimat_right<MAX_RIGHT_MATRICES_PER_THREAD><<<num_blocks, num_threads>>>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices
            );

            ccn_shuffle_multimat_right_kernel_args::record_launch(
                num_threads,
                num_blocks,
                MAX_RIGHT_MATRICES_PER_THREAD
            );
        } else {
            ccn_shuffle_multimat_right_right_mats_dispatch<MAX_RIGHT_MATRICES_PER_THREAD - 1>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                warps_per_thread_block,
                right_matrices_per_thread
            );
        }
    }
}

/**
 * For description of the functionality implemented by this kernel, see ccn_shuffle_multimat_right kernel.
 * This kernel adds distribution of rows of a single shift between multiple threads.
 *
 * @tparam T
 * @tparam RES
 * @param left
 * @param right
 * @param out
 * @param matrix_size
 * @param search_size
 */
template<dsize_t MAX_RIGHT_MATRICES_PER_THREAD, typename DIST, typename T, typename RES>
__global__ void ccn_shuffle_multimat_right_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t max_rows_per_thread
) {

    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);


    dsize_t warp_output_x_offset;
    // Index of the first matrix in the group processed by the current thread
    dsize_t matrix_group_start_idx;
    get_matrix_group(
        search_size.x,
        ctb.group_index().x,
        MAX_RIGHT_MATRICES_PER_THREAD,
        warp_output_x_offset,
        matrix_group_start_idx
    );

    // Distribute rows of a single shift between multiple workers,
    // in this case threads
    // Return the assigned output row (which corresponds to a shift),
    // together with the number of workers computing this shift and
    // index of the current worker in range [0, number_of_workers_for_shift)
    assigned_work work = DIST::distribute_rows(
        ctb.group_index().y * ctb.group_dim().y + ctb.thread_index().y,
        max_rows_per_thread,
        matrix_size.y,
        search_size.y
    );

    // All threads of a warp should share the same worker_idx and workers_for_row
    // so either the whole warp continues or exists
    if (work.worker_idx >= work.workers_for_row) {
        return;
    }

    // All warps of given block start at the same x, but each work on different row of output
    dsize2_t thread0_out_pos{
        warp_output_x_offset,
        work.output_row
    };
    dsize2_t last_warp_thread_out_pos = thread0_out_pos +
                                        dsize2_t{warp.size() - 1, 0};

    // Position in the output matrix
    // This is unique for each thread, as each thread computes a single shift which
    // corresponds to a single output value
    dsize2_t output_pos = thread0_out_pos +
                          dsize2_t{warp.thread_rank(), 0};

    dsize2_t half_search_size = (search_size - 1) / 2;

    // Min of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 0
    vec2<int> warp_min_shift{
        static_cast<int>(thread0_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(thread0_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    // Max of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 31
    // It is clamped into search size as matrix may not be of size divisible by warp_size
    vec2<int> warp_max_shift{
        static_cast<int>(min(last_warp_thread_out_pos.x, search_size.x)) - static_cast<int>(half_search_size.x),
        static_cast<int>(min(last_warp_thread_out_pos.y, search_size.y)) - static_cast<int>(half_search_size.y)
    };


    // The start depends on the how far right the right matrix is shifted over the left matrix
    // if the right most shift, aka max shift is positive, that means that the left side of the right
    // matrix is inside the left matrix, so we need to start from the 0 element
    // if the max shift is negative, then absolute value tells us how many items of the right matrix are not needed
    // as they do not overlap in any shift computed by the matrix, as all smaller shifts have the right matrix more to the left
    // so they overlap less values
    dsize_t warp_x_right_start = warp_max_shift.x >= 0 ? 0 : -warp_max_shift.x;

    // The last value will be read by the min shift, so if it is larger than 0, the right side of the right matrix overhangs
    // the left matrix and so we don't need to reed the last abs(min_shift) values. Otherwise the right side of the right
    // matrix is inside the left matrix and we need to read it till the end.
    dsize_t warp_x_right_end = warp_min_shift.x >= 0 ? matrix_size.x - warp_min_shift.x : matrix_size.x;

    // All threads in a warp process the same range of rows, so warp_min_shift.y and warp_max_shift.y are the same
    // Multiple threads from different warps may compute the same shift
    // These values are shared for all workers computing the same shift
    dsize_t shared_y_right_start = max(-warp_min_shift.y, 0);
    dsize_t shared_y_right_end = min(matrix_size.y - warp_max_shift.y, matrix_size.y);

    dsize_t shared_overlapping_rows = shared_y_right_end - shared_y_right_start;
    dsize_t rows_per_worker = div_up(shared_overlapping_rows, work.workers_for_row);


    // For the current worker
    dsize_t warp_y_right_start = shared_y_right_start + work.worker_idx * rows_per_worker;
    dsize_t warp_y_right_end = min(warp_y_right_start + rows_per_worker, shared_y_right_end);


    dsize_t thread_num_right_matrices = min(num_right_matrices - matrix_group_start_idx, MAX_RIGHT_MATRICES_PER_THREAD);

    auto args = create_warp_shuffle_impl_args(
        left,
        right + matrix_group_start_idx * matrix_size.area(),
        out + matrix_group_start_idx * search_size.area(),
        dsize2_t{warp_x_right_start, warp_y_right_start},
        dsize2_t{warp_x_right_end, warp_y_right_end},
        warp_min_shift,
        output_pos,
        matrix_size,
        search_size
    );

    warp_shuffle_impl_right_mats_dispatch<MAX_RIGHT_MATRICES_PER_THREAD, true>(
        warp,
        thread_num_right_matrices,
        args
    );
}

/**
 * Args used for the kernel call. The class is a singleton to minimize the impact
 * on measured time (prevent allocation etc.)
 */
class ccn_shuffle_multimat_right_work_distribution_kernel_args : public kernel_args {
public:
    dsize_t max_right_matrices_per_thread_;
    distribution dist_;

    ccn_shuffle_multimat_right_work_distribution_kernel_args(const ccn_shuffle_multimat_right_work_distribution_kernel_args&) = delete;
    ccn_shuffle_multimat_right_work_distribution_kernel_args& operator=(ccn_shuffle_multimat_right_work_distribution_kernel_args&) = delete;

    static void record_launch(
        dim3 block_size,
        dim3 grid_size,
        dsize_t max_right_matrices_per_thread,
        distribution dist
    ) {
        static ccn_shuffle_multimat_right_work_distribution_kernel_args instance;
        instance.set_common(block_size, grid_size, 0);
        instance.max_right_matrices_per_thread_ = max_right_matrices_per_thread;
        instance.dist_ = dist;
        set_last_kernel_launch_args(&instance);
    }

    [[nodiscard]] std::unordered_map<std::string, std::string> get_additional_args() const override {
        return std::unordered_map<std::string, std::string>{
            {"max_right_matrices_per_thread", std::to_string(max_right_matrices_per_thread_)},
            {"work_distribution", to_string(dist_)}
        };
    }

private:
    ccn_shuffle_multimat_right_work_distribution_kernel_args()
        : kernel_args(),
          max_right_matrices_per_thread_(0),
          dist_(distribution::none)
    { }
};

template<dsize_t MAX_RIGHT_MATRICES_PER_THREAD, typename DIST, typename T, typename RES>
__host__ void ccn_shuffle_multimat_right_work_distribution_right_mats_dispatch(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread,
    cudaStream_t cuda_stream
) {
    if constexpr(MAX_RIGHT_MATRICES_PER_THREAD == 0) {
        // Silence the unused parameter warning
        (void)left;
        (void)right;
        (void)out;
        (void)matrix_size;
        (void)search_size;
        (void)num_right_matrices;
        (void)warps_per_thread_block;
        (void)right_matrices_per_thread;
        (void)max_rows_per_thread;
        (void)cuda_stream;
        assert(false);
    } else {
        if (MAX_RIGHT_MATRICES_PER_THREAD == right_matrices_per_thread) {
            dsize_t num_workers = DIST::num_workers(max_rows_per_thread, matrix_size.y, search_size.y);

            // Each row of cuda block corresponds to a single warp for simplified code
            constexpr dsize_t block_x_size = warp_size;

            dsize_t num_matrix_groups = div_up(num_right_matrices, right_matrices_per_thread);
            dsize_t blocks_per_matrix_group = div_up(search_size.x, block_x_size);

            dim3 num_threads(block_x_size, warps_per_thread_block);
            dim3 num_blocks(
                blocks_per_matrix_group * num_matrix_groups,
                div_up(num_workers, num_threads.y)
            );

            ccn_shuffle_multimat_right_work_distribution<MAX_RIGHT_MATRICES_PER_THREAD, DIST><<<num_blocks, num_threads, 0, cuda_stream>>>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                max_rows_per_thread
            );

            ccn_shuffle_multimat_right_work_distribution_kernel_args::record_launch(
                num_threads,
                num_blocks,
                MAX_RIGHT_MATRICES_PER_THREAD,
                DIST::type
            );
        } else {
            ccn_shuffle_multimat_right_work_distribution_right_mats_dispatch<MAX_RIGHT_MATRICES_PER_THREAD - 1, DIST>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                warps_per_thread_block,
                right_matrices_per_thread,
                max_rows_per_thread,
                cuda_stream
            );
        }
    }
}

} // END anonymous namespace

template<typename T, typename RES>
void run_ccn_shuffle_multimat_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread
) {
    if (warps_per_thread_block > 32) {
        throw std::runtime_error("Too many warps per thread block: "s + std::to_string(warps_per_thread_block) + " (max 32)");
    }

    if (right_matrices_per_thread == 0 || right_matrices_per_thread > right_matrices_per_thread_limit) {
        throw std::runtime_error(
            "Invalid number of right matrices per thread: "s +
            std::to_string(right_matrices_per_thread) +
            " [1-"s +
            std::to_string(right_matrices_per_thread_limit) +
            "]"s
        );
    }

    ccn_shuffle_multimat_right_right_mats_dispatch<right_matrices_per_thread_limit>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        num_right_matrices,
        warps_per_thread_block,
        right_matrices_per_thread
    );
}

template<typename DIST, typename T, typename RES>
void run_ccn_shuffle_multimat_right_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread,
    cudaStream_t cuda_stream
) {
    if (warps_per_thread_block > 32) {
        throw std::runtime_error("Too many warps per thread block: "s + std::to_string(warps_per_thread_block) + " (max 32)");
    }

    if (right_matrices_per_thread > right_matrices_per_thread_limit) {
        throw std::runtime_error("Too many right matrices per thread: "s +
                                 std::to_string(right_matrices_per_thread) +
                                 " (max "s +
                                 std::to_string(right_matrices_per_thread_limit) +
                                 ")"s
        );
    }

    ccn_shuffle_multimat_right_work_distribution_right_mats_dispatch<right_matrices_per_thread_limit, DIST>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        num_right_matrices,
        warps_per_thread_block,
        right_matrices_per_thread,
        max_rows_per_thread,
        cuda_stream
    );
}

// template void run_ccn_shuffle_multimat_right<int, int>(
//         const int* __restrict__ left,
//         const int* __restrict__ right,
//         int* __restrict__ out,
//         dsize2_t matrix_size,
//         dsize2_t search_size,
//         dsize_t num_right_matrices,
//         dsize_t warps_per_thread_block,
//         dsize_t right_matrices_per_thread
// );

template void run_ccn_shuffle_multimat_right<float, float>(
        const float* __restrict__ left,
        const float* __restrict__ right,
        float* __restrict__ out,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t num_right_matrices,
        dsize_t warps_per_thread_block,
        dsize_t right_matrices_per_thread
);

// template void run_ccn_shuffle_multimat_right<double, double>(
//         const double* __restrict__ left,
//         const double* __restrict__ right,
//         double* __restrict__ out,
//         dsize2_t matrix_size,
//         dsize2_t search_size,
//         dsize_t num_right_matrices,
//         dsize_t warps_per_thread_block,
//         dsize_t right_matrices_per_thread
// );

// template void run_ccn_shuffle_multimat_right_work_distribution<triangle_distribution, int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t right_matrices_per_thread,
//     dsize_t max_rows_per_thread,
//     cudaStream_t cuda_stream
// );

template void run_ccn_shuffle_multimat_right_work_distribution<triangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread,
    cudaStream_t cuda_stream
);

// template void run_ccn_shuffle_multimat_right_work_distribution<triangle_distribution, double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t right_matrices_per_thread,
//     dsize_t max_rows_per_thread,
//     cudaStream_t cuda_stream
// );

// template void run_ccn_shuffle_multimat_right_work_distribution<rectangle_distribution, int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t right_matrices_per_thread,
//     dsize_t max_rows_per_thread,
//     cudaStream_t cuda_stream
// );

template void run_ccn_shuffle_multimat_right_work_distribution<rectangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread,
    cudaStream_t cuda_stream
);

// template void run_ccn_shuffle_multimat_right_work_distribution<rectangle_distribution, double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t right_matrices_per_thread,
//     dsize_t max_rows_per_thread,
//     cudaStream_t cuda_stream
// );

// template void run_ccn_shuffle_multimat_right_work_distribution<no_distribution, int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t right_matrices_per_thread,
//     dsize_t max_rows_per_thread,
//     cudaStream_t cuda_stream
// );

template void run_ccn_shuffle_multimat_right_work_distribution<no_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread,
    cudaStream_t cuda_stream
);

// template void run_ccn_shuffle_multimat_right_work_distribution<no_distribution, double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t right_matrices_per_thread,
//     dsize_t max_rows_per_thread,
//     cudaStream_t cuda_stream
// );

}
