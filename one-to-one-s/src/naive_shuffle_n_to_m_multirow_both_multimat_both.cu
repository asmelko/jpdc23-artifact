#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <stdexcept>
#include <cassert>

#include "types.cuh"
#include "cuda_helpers.cuh"
#include "bound_checked_loads.cuh"

#include "warp_size.hpp"
#include "kernel_args.hpp"

namespace cg = cooperative_groups;

namespace cross {

namespace {

constexpr dsize_t shifts_per_thread_per_right_matrix_limit = SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_SHIFTS_PER_THREAD_PER_RIGHT_MATRIX_LIMIT;
constexpr dsize_t right_matrices_per_thread_limit = SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT;
constexpr dsize_t left_matrices_per_thread_limit = SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT;
constexpr dsize_t left_rows_per_iteration_limit = SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_ROWS_PER_ITERATION_LIMIT;

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
    vec2<int> warp_max_shift;
    dsize2_t output_pos;
    dsize2_t matrix_size;
    dsize2_t search_size;
    dsize_t num_right_matrices;

    __device__ warp_shuffle_impl_args(
        const T* __restrict__ left,
        const T* __restrict__ right,
        RES* __restrict__ out,
        dsize2_t warp_right_start,
        dsize2_t warp_right_end,
        vec2<int> warp_min_shift,
        vec2<int> warp_max_shift,
        dsize2_t output_pos,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t num_right_matrices
    ) : left(left), right(right), out(out), warp_right_start(warp_right_start),
        warp_right_end(warp_right_end), warp_min_shift(warp_min_shift), warp_max_shift(warp_max_shift),
        output_pos(output_pos), matrix_size(matrix_size), search_size(search_size),
        num_right_matrices(num_right_matrices) {

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
    vec2<int> warp_max_shift,
    dsize2_t output_pos,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices
) {
    return warp_shuffle_impl_args<T, RES>(
        left,
        right,
        out,
        warp_right_start,
        warp_right_end,
        warp_min_shift,
        warp_max_shift,
        output_pos,
        matrix_size,
        search_size,
        num_right_matrices
    );
}

template<dsize_t NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t NUM_LEFT_ROWS, dsize_t MAX_NUM_SHIFTS_PER_RIGHT_MAT, dsize_t SUM_START, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void compute_row_group(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    dsize_t warp_y_right_start,
    int y_shift,
    RES (&sum)[MAX_NUM_SHIFTS_PER_RIGHT_MAT * NUM_RIGHT_MATS * NUM_LEFT_MATS]
) {
    dsize_t warp_y_left = warp_y_right_start + y_shift;
    const T* first_left_row = args.left + warp_y_left * args.matrix_size.x;

    const dsize_t first_right_row_offset = warp_y_right_start * args.matrix_size.x;
    const T* first_right_row = args.right + first_right_row_offset;

    int warp_x_left = static_cast<int>(args.warp_right_start.x) + args.warp_min_shift.x;

    constexpr dsize_t TOTAL_LEFT_ROWS = NUM_LEFT_ROWS * NUM_LEFT_MATS;
    // Preload the first values from left matrix
    T thread_left_bottom[TOTAL_LEFT_ROWS];
    #pragma unroll
    for (dsize_t l_mat = 0; l_mat < NUM_LEFT_MATS; ++l_mat) {
        #pragma unroll
        for (dsize_t l_row = 0; l_row < NUM_LEFT_ROWS; ++l_row) {
            thread_left_bottom[l_mat * NUM_LEFT_ROWS + l_row] = load_with_bounds_check(
                first_left_row + l_mat * args.matrix_size.area() + l_row * args.matrix_size.x,
                warp_x_left + warp.thread_rank(),
                args.matrix_size.x
            );
        }
    }

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

        // We need this many right values because first left row
        // is computed with rows 0 to NUM_SHIFTS_PER_MAT - 1, second left row is computed
        // with rows 1 to NUM_SHIFTS_PER_MAT, third left row with 2 to NUM_SHIFTS_PER_MAT + 1
        constexpr dsize_t NUM_RIGHT_ROWS = NUM_SHIFTS_PER_RIGHT_MAT + NUM_LEFT_ROWS - 1;
        constexpr dsize_t TOTAL_RIGHT_ROWS = NUM_RIGHT_ROWS * NUM_RIGHT_MATS;
        // Load values from num_rights right matrices
        T thread_right[TOTAL_RIGHT_ROWS];
        #pragma unroll
        for (dsize_t r_mat = 0; r_mat < NUM_RIGHT_MATS; ++r_mat) {
            #pragma unroll
            for (dsize_t r_row = 0; r_row < NUM_RIGHT_ROWS; ++r_row) {
                thread_right[r_mat * NUM_RIGHT_ROWS + r_row] = load_with_bounds_check(
                    first_right_row + r_mat * args.matrix_size.area() + r_row * args.matrix_size.x,
                    right_idx,
                    args.matrix_size.x
                );
            }
        }

        T thread_left_top[TOTAL_LEFT_ROWS];
        #pragma unroll
        for (dsize_t l_mat = 0; l_mat < NUM_LEFT_MATS; ++l_mat) {
            for (dsize_t l_row = 0; l_row < NUM_LEFT_ROWS; ++l_row) {
                thread_left_top[l_mat * NUM_LEFT_ROWS + l_row] = load_with_bounds_check(
                    first_left_row + l_mat * args.matrix_size.area() + l_row * args.matrix_size.x,
                    left_idx,
                    args.matrix_size.x
                );
            }
        }

        // TODO: Maybe pragma unroll?
        for (dsize_t i = 0; i < warp.size(); ++i) {
            #pragma unroll
            for (dsize_t r_mat = 0; r_mat < NUM_RIGHT_MATS; ++r_mat) {
                #pragma unroll
                for (dsize_t r_row = 0; r_row < NUM_RIGHT_ROWS; ++r_row) {
                    // Broadcast
                    auto right_val = warp.shfl(thread_right[r_mat * NUM_RIGHT_ROWS + r_row], i);


                    #pragma unroll
                    for (dsize_t l_mat = 0; l_mat < NUM_LEFT_MATS; ++l_mat) {
                        #pragma unroll
                        for (dsize_t l_row = 0; l_row < NUM_LEFT_ROWS; ++l_row) {
                            // Some combinations are not valid, as described by the NUM_RIGHT_ROWS
                            // variable comment.
                            // left row 0 is computed with right rows 0 to NUM_SHIFTS_PER_MAT - 1
                            // left row 1 is computed with right rows 1 to NUM_SHIFTS_PER_MAT
                            // left row 2 is computed with right rows 2 to NUM_SHIFTS_PER_MAT + 1
                            // TODO: Try if using break or continue can still be unrolled
                            if (l_row <= r_row && r_row < NUM_SHIFTS_PER_RIGHT_MAT + l_row) {
                                // As we need to offset the sum by SUM_START exluding given shifts from ALL matrices
                                // we need to order it so that first are shifts 0 from all matrices,
                                // then shifts 1 from all matrices etc.
                                // so first shift 0 from all right matrices against first left matrix
                                // then shift 0 from all right matrices against second left matrix etc.
                                // then shift 1 from all right matrices against first left matrix etc.
                                // This is why the shift, computed as described below, must be multiplied by NUM_RIGHT_MATS * NUM_LEFT_MATS
                                // (r_row - l_row) gets us the right row index compared to the left row
                                // (NUM_SHIFTS_PER_RIGHT_MAT - 1 - (r_row - l_row)) as the rows from the right matrix are loaded top to bottom
                                // but as we compute them against last row from the left matrix they overlap with,
                                // the row 0 from the right matrix overlaps with the given row from the left matrix
                                // in overlap NUM_RIGHT_ROWS - 1 etc.
                                sum[SUM_START + (NUM_SHIFTS_PER_RIGHT_MAT - 1 - (r_row - l_row)) * NUM_RIGHT_MATS * NUM_LEFT_MATS + l_mat * NUM_RIGHT_MATS + r_mat] +=
                                    thread_left_bottom[l_mat * NUM_LEFT_ROWS + l_row] * right_val;
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (dsize_t l = 0; l < TOTAL_LEFT_ROWS; ++l) {

                // This if cannot be changed into ternary operator
                // as nvcc fails to optimize the two arrays into registers
                // and instead puts them into local memory when ternary operator
                // is used
                T bottom_shift_val;
                if (warp.thread_rank() != 0) {
                    bottom_shift_val = thread_left_bottom[l];
                } else {
                    // Lane 0 pushes the bottom-most value of the top buffer to the top of the bottom buffer
                    //  making it behave as one continuous buffer
                    bottom_shift_val = thread_left_top[l];
                }
                // Shuffle does modulo srcLane automatically
                thread_left_bottom[l] = warp.shfl(bottom_shift_val, warp.thread_rank() + 1);

                thread_left_top[l] = warp.shfl_down(thread_left_top[l], 1);
            }
        }
    }
}

/*
 * First NUM_RIGHT_ROWS rows will only overlap in some of the shifts
 * If we start at the 0 row of the right matrix, then that means that the
 * top of the right matrix is inside the left matrix
 *
 * As we are computing NUM_RIGHT_ROWS shifts in consecutive rows with the same
 * x coordinate, the first shift will overlap given left row and no other shift
 * overlaps anything with the left row
 *
 * Next left row is overlapped with the args.warp_right_start.y by the following shift,
 * while the first shift overlaps the left row with args.warp_right_start.y + 1
 *
 * Then the third left row is overlapped with args.warp_right_start.y by the third shift,
 * with args.warp_right_start.y + 1 by second shift and with args.warp_right_start.y + 2 by
 * first shift etc.
 *
 * If the top of the right matrix starts outside the left matrix, which can only be above the
 * left matrix, some of the steps may be skipped, for example if it is one row above,
 * the first left row is overlapped by the first shift with row args.warp_right_start.y + 1
 * and by the second shift with row args.warp_right_start.y, which is exactly the second step described above
 *
 * Similar principle, but in reverse, applies when bottom of the right matrix is inside the left matrix.
 * There the left row stays the same, but we change the number of right rows it runs against,
 * getting progressively smaller.
 *
 * These ifs should cover all possibilities up to NUM_RIGHT_ROWS
 * Because max_shift.y - min_shift.y == NUM_RIGHT_ROWS, min_shift.y + NUM_RIGHT_ROWS == max_shift.y
 *
 */
template<int NUM_SHIFTS_PER_RIGHT_MAT, dsize_t MAX_NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void startup(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    RES (&sum)[MAX_NUM_SHIFTS_PER_RIGHT_MAT * NUM_RIGHT_MATS * NUM_LEFT_MATS]
) {
    if constexpr(NUM_SHIFTS_PER_RIGHT_MAT < MAX_NUM_SHIFTS_PER_RIGHT_MAT) {
        if (static_cast<int>(args.warp_right_start.y) + args.warp_min_shift.y + NUM_SHIFTS_PER_RIGHT_MAT - 1 >= 0) {
            compute_row_group<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, 1, MAX_NUM_SHIFTS_PER_RIGHT_MAT, 0>(
                warp,
                args,
                args.warp_right_start.y,
                args.warp_min_shift.y + NUM_SHIFTS_PER_RIGHT_MAT - 1,
                sum
            );
        }
        startup<NUM_SHIFTS_PER_RIGHT_MAT + 1, MAX_NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS>(warp, args, sum);
    } else {
        // Silence the unused parameter warning
        (void)warp;
        (void)args;
        (void)sum;
    }
}

template<int NUM_SHIFTS_PER_RIGHT_MAT, dsize_t MAX_NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void wind_down(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    RES (&sum)[MAX_NUM_SHIFTS_PER_RIGHT_MAT * NUM_RIGHT_MATS * NUM_LEFT_MATS]
) {
    if constexpr(NUM_SHIFTS_PER_RIGHT_MAT > 0) {
        if (args.warp_right_end.y - NUM_SHIFTS_PER_RIGHT_MAT + args.warp_max_shift.y < args.matrix_size.y) {
            // The SUM_START is set to exclude the first MAX_NUM_SHIFTS_PER_RIGHT_MAT - NUM_SHIFTS_PER_RIGHT_MAT shifts
            // As sum is ordered shift_0_lmat_0_rmat_0, shift_0_lmat_0_rmat_1 ... shift_0_lmat_1_rmat_0 ... shift_1_lmat_0_rmat_0
            compute_row_group<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, 1, MAX_NUM_SHIFTS_PER_RIGHT_MAT, (MAX_NUM_SHIFTS_PER_RIGHT_MAT - NUM_SHIFTS_PER_RIGHT_MAT) * NUM_LEFT_MATS * NUM_RIGHT_MATS>(
                warp,
                args,
                args.warp_right_end.y - NUM_SHIFTS_PER_RIGHT_MAT,
                args.warp_max_shift.y,
                sum
            );
        }
        wind_down<NUM_SHIFTS_PER_RIGHT_MAT - 1, MAX_NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS>(warp, args, sum);
    } else {
        // Silence the unused parameter warning
        (void)warp;
        (void)args;
        (void)sum;
    }
}


template<dsize_t NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t LEFT_ROWS_PER_ITER, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void n_to_m_shuffle_multirow_both_multimat_both_impl(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    const warp_shuffle_impl_args<T, RES>& args
) {
    // Sum is ordered shift_0_lmat_0_rmat_0, shift_0_lmat_0_rmat_1 ... shift_0_lmat_1_rmat_0 ... shift_1_lmat_0_rmat_0
    // So that we can exclude first x shifts from all matrices for wind_down
    T sum[NUM_SHIFTS_PER_RIGHT_MAT * NUM_RIGHT_MATS * NUM_LEFT_MATS];
    for (dsize_t s = 0; s < NUM_SHIFTS_PER_RIGHT_MAT * NUM_RIGHT_MATS * NUM_LEFT_MATS; ++s) {
        sum[s] = 0;
    }

    startup<1, NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS>(warp, args, sum);

    /*
     * The startup gets us to the situation where we have the first
     * left row at max_shift (== min_shift + NUM_RIGHTS_ROW) which is
     * to be processed with all NUM_RIGHT_ROWS
     * As we are always loading warp_y_right and the following (NUM_THREAD_SHIFTS + MAX_LEFT_ROWS - 1) rows,
     * we need to stop NUM_THREAD_SHIFTS + MAX_LEFT_ROWS - 1 before the end
     */
    int multileft_end = args.warp_right_end.y - (NUM_SHIFTS_PER_RIGHT_MAT + LEFT_ROWS_PER_ITER - 1);
    int warp_y_right = args.warp_right_start.y;
    for (; warp_y_right < multileft_end; warp_y_right += LEFT_ROWS_PER_ITER) {
        compute_row_group<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, LEFT_ROWS_PER_ITER, NUM_SHIFTS_PER_RIGHT_MAT, 0>(
            warp,
            args,
            warp_y_right,
            args.warp_max_shift.y,
            sum
        );
    }

    /*
     * Finish the possible MAX_LEFT_ROWS - 1 left rows left over before the original wind-down
     * As we are always loading warp_y_right and the following (NUM_THREAD_SHIFTS - 1) rows,
     * we need to stop NUM_THREAD_SHIFTS before the end
     * TODO: Try template generated if tree that will use just one call with the correct number of left rows
     */
    int total_end = args.warp_right_end.y - (NUM_SHIFTS_PER_RIGHT_MAT - 1);
    for (; warp_y_right < total_end; warp_y_right += 1) {
        compute_row_group<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, 1, NUM_SHIFTS_PER_RIGHT_MAT, 0>(
            warp,
            args,
            warp_y_right,
            args.warp_max_shift.y,
            sum
        );
    }

    wind_down<NUM_SHIFTS_PER_RIGHT_MAT - 1, NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS>(warp, args, sum);

    auto first_output_offset = args.output_pos.linear_idx(args.search_size.x);
    RES* matrix = args.out;

    // TODO: Maybe just check the x axis, Y axis should be filtered out by 0 NUM_RIGHT_ROWS
    if (args.output_pos.x < args.search_size.x && args.output_pos.y < args.search_size.y) {
        for (dsize_t l_mat = 0; l_mat < NUM_LEFT_MATS; ++l_mat) {
            for (dsize_t r_mat = 0; r_mat < NUM_RIGHT_MATS; ++r_mat) {
                for (dsize_t shift = 0; shift < NUM_SHIFTS_PER_RIGHT_MAT; ++shift) {
                    auto output_offset = first_output_offset + (l_mat * args.num_right_matrices + r_mat) * args.search_size.area() + shift * args.search_size.x;
                    auto val = sum[(shift * NUM_LEFT_MATS * NUM_RIGHT_MATS + l_mat * NUM_RIGHT_MATS + r_mat)];
                    if constexpr(ATOMIC) {
                        atomicAdd(matrix + output_offset, val);
                    } else {
                        matrix[output_offset] = val;
                    }
                }
            }
        }
    }
}

template<dsize_t NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t LEFT_ROWS_PER_ITER, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void n_to_m_shuffle_multirow_both_multimat_both_impl_left_mats_dispatch(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t num_left_mats,
    const warp_shuffle_impl_args<T, RES>& args
) {
    if constexpr(NUM_LEFT_MATS == 0) {
        // Silence the unused parameter warning
        (void)warp;
        (void)num_left_mats;
        (void)args;
        assert(false);
    } else {
        if (NUM_LEFT_MATS == num_left_mats) {
            n_to_m_shuffle_multirow_both_multimat_both_impl<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, LEFT_ROWS_PER_ITER, ATOMIC>(
                warp,
                args
            );
        } else {
            n_to_m_shuffle_multirow_both_multimat_both_impl_left_mats_dispatch<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS - 1, LEFT_ROWS_PER_ITER, ATOMIC>(
                warp,
                num_left_mats,
                args
            );
        }
    }
}

template<dsize_t NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t LEFT_ROWS_PER_ITER, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void n_to_m_shuffle_multirow_both_multimat_both_impl_right_mats_dispatch(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t num_left_mats,
    dsize_t num_right_mats,
    const warp_shuffle_impl_args<T, RES>& args
) {
    if constexpr(NUM_RIGHT_MATS == 0) {
        // Silence the unused parameter warning
        (void)warp;
        (void)num_left_mats;
        (void)num_right_mats;
        (void)args;
        assert(false);
    } else {
        if (NUM_RIGHT_MATS == num_right_mats) {
            n_to_m_shuffle_multirow_both_multimat_both_impl_left_mats_dispatch<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, LEFT_ROWS_PER_ITER, ATOMIC>(
                warp,
                num_left_mats,
                args
            );
        } else {
            n_to_m_shuffle_multirow_both_multimat_both_impl_right_mats_dispatch<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS - 1, NUM_LEFT_MATS, LEFT_ROWS_PER_ITER, ATOMIC>(
                warp,
                num_left_mats,
                num_right_mats,
                args
            );
        }
    }
}

template<dsize_t NUM_SHIFTS_PER_RIGHT_MAT, dsize_t NUM_RIGHT_MATS, dsize_t NUM_LEFT_MATS, dsize_t LEFT_ROWS_PER_ITER, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void n_to_m_shuffle_multirow_both_multimat_both_impl_shifts_dispatch(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t num_shifts_per_right_mat,
    dsize_t num_left_mats,
    dsize_t num_right_mats,
    const warp_shuffle_impl_args<T, RES>& args
) {
    if constexpr(NUM_SHIFTS_PER_RIGHT_MAT == 0) {
        // Zero is a valid value, which happens when search_size is not divisible by
        // block_size.z with remainder greater than 3

        // Silence the unused parameter warning
        (void)warp;
        (void)num_shifts_per_right_mat;
        (void)num_left_mats;
        (void)num_right_mats;
        (void)args;
    } else {
        if (NUM_SHIFTS_PER_RIGHT_MAT == num_shifts_per_right_mat) {
            n_to_m_shuffle_multirow_both_multimat_both_impl_right_mats_dispatch<NUM_SHIFTS_PER_RIGHT_MAT, NUM_RIGHT_MATS, NUM_LEFT_MATS, LEFT_ROWS_PER_ITER, ATOMIC>(
                warp,
                num_left_mats,
                num_right_mats,
                args
            );
        } else {
            n_to_m_shuffle_multirow_both_multimat_both_impl_shifts_dispatch<NUM_SHIFTS_PER_RIGHT_MAT - 1, NUM_RIGHT_MATS, NUM_LEFT_MATS, LEFT_ROWS_PER_ITER, ATOMIC>(
                warp,
                num_shifts_per_right_mat,
                num_left_mats,
                num_right_mats,
                args
            );
        }
    }
}

/**
 * For description of the functionality implemented by this kernel, see ccn_warp_shuffle kernel.
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
template<dsize_t MAX_SHIFTS_PER_RIGHT_MATRIX, dsize_t MAX_RIGHT_MATRICES_PER_THREAD, dsize_t MAX_LEFT_MATRICES_PER_THREAD, dsize_t LEFT_ROWS_PER_ITER, typename T, typename RES>
__global__ void ccn_n_to_m_shuffle_multirow_both_multimat_both(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices
) {

    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    // Matrix group is the group of right matrices (of at most right_matrices_per_thread matrices)
    // for which current thread computes the given shift
    // All warps in a block process the same 32 shifts in the x axis, but on different rows
    // so warps in the first block compute shifts 0-31, warps in the second block compute shifts 32-63 etc.
    // So each matrix_group needs to have as many threads as there are shifts in the x axis
    // so number of shifts in the x axis / warp_size
    // TODO: This is precomputed on CPU so we could pass it from there
    dsize_t blocks_per_matrix_group = div_up(search_size.x, warp_size);

    // Which matrix group this block and all its warps will compute
    // THe X axis of block index encodes the shift in x axis together with the left matrix group the thread belongs to
    dsize_t left_matrix_group_idx = ctb.group_index().x / blocks_per_matrix_group;
    dsize_t right_matrix_group_idx = ctb.group_index().y;

    // Offset of the current block and all of its warps in its matrix group
    // This corresponds to the position to write to in the output and the shift
    // to compute
    dsize_t matrix_group_block_offset = ctb.group_index().x % blocks_per_matrix_group;
    dsize_t warp_output_x_offset = matrix_group_block_offset * warp_size;

    // Index of the first matrix in the group processed by the current thread
    dsize_t left_matrix_group_start_idx = left_matrix_group_idx * MAX_LEFT_MATRICES_PER_THREAD;
    dsize_t right_matrix_group_start_idx = right_matrix_group_idx * MAX_RIGHT_MATRICES_PER_THREAD;

    // All warps of given block start at the same x, but each work on different row of output
    dsize2_t thread0_out_pos{
        warp_output_x_offset,
        (ctb.group_index().z * ctb.group_dim().y + ctb.thread_index().y) * MAX_SHIFTS_PER_RIGHT_MATRIX
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
        static_cast<int>(min(last_warp_thread_out_pos.x, search_size.x - 1)) -
            static_cast<int>(half_search_size.x),
        static_cast<int>(min(last_warp_thread_out_pos.y + MAX_SHIFTS_PER_RIGHT_MATRIX - 1, search_size.y - 1)) -
            static_cast<int>(half_search_size.y)
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

    dsize_t warp_y_right_start = max(-warp_max_shift.y, 0);
    dsize_t warp_y_right_end = min(matrix_size.y - warp_min_shift.y, matrix_size.y);

    // Max shift might be smaller than min shift if warp is completely outside the out matrix
    // +1 because max_shift is inclusive, it is the last shift computed by this warp
    // so to get the number of shifts with both sides inclusive, we need to add 1
    auto warp_num_shifts_per_right_mat = static_cast<dsize_t>(max(warp_max_shift.y - warp_min_shift.y + 1, 0));
    dsize_t warp_num_left_matrices = min(
        num_left_matrices - left_matrix_group_start_idx, MAX_LEFT_MATRICES_PER_THREAD
    );
    dsize_t warp_num_right_matrices = min(
        num_right_matrices - right_matrix_group_start_idx, MAX_RIGHT_MATRICES_PER_THREAD
    );

    auto args = create_warp_shuffle_impl_args(
        left + left_matrix_group_start_idx * matrix_size.area(),
        right + right_matrix_group_start_idx * matrix_size.area(),
        out + (left_matrix_group_start_idx * num_right_matrices + right_matrix_group_start_idx) * search_size.area(),
        dsize2_t{warp_x_right_start, warp_y_right_start},
        dsize2_t{warp_x_right_end, warp_y_right_end},
        warp_min_shift,
        warp_max_shift,
        output_pos,
        matrix_size,
        search_size,
        num_right_matrices
    );

    n_to_m_shuffle_multirow_both_multimat_both_impl_shifts_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER, false>(
        warp,
        warp_num_shifts_per_right_mat,
        warp_num_left_matrices,
        warp_num_right_matrices,
        args
    );
}

/**
 * Args used for the kernel call. The class is a singleton to minimize the impact
 * on measured time (prevent allocation etc.)
 */
class ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args : public kernel_args {
public:
    dsize_t max_shifts_per_right_matrix_;
    dsize_t max_right_matrices_per_thread_;
    dsize_t max_left_matrices_per_thread_;
    dsize_t left_rows_per_iter_;

    ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args(const ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args&) = delete;
    ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args& operator=(ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args&) = delete;

    static void record_launch(
        dim3 block_size,
        dim3 grid_size,
        dsize_t max_shifts_per_right_matrix,
        dsize_t max_right_matrices_per_thread,
        dsize_t max_left_matrices_per_thread,
        dsize_t left_rows_per_iter
    ) {
        static ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args instance;
        instance.set_common(block_size, grid_size, 0);
        instance.max_shifts_per_right_matrix_ = max_shifts_per_right_matrix;
        instance.max_right_matrices_per_thread_ = max_right_matrices_per_thread;
        instance.max_left_matrices_per_thread_ = max_left_matrices_per_thread;
        instance.left_rows_per_iter_ = left_rows_per_iter;
        set_last_kernel_launch_args(&instance);
    }

    [[nodiscard]] std::unordered_map<std::string, std::string> get_additional_args() const override {
        return std::unordered_map<std::string, std::string>{
            {"max_shifts_per_right_matrix", std::to_string(max_shifts_per_right_matrix_)},
            {"max_right_matrices_per_thread", std::to_string(max_right_matrices_per_thread_)},
            {"max_left_matrices_per_thread", std::to_string(max_left_matrices_per_thread_)},
            {"left_rows_per_iter", std::to_string(left_rows_per_iter_)}
        };
    }

private:
    ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args()
    : kernel_args(),
        max_shifts_per_right_matrix_(0),
        max_right_matrices_per_thread_(0),
        max_left_matrices_per_thread_(0),
        left_rows_per_iter_(0)
    { }
};

template<dsize_t MAX_SHIFTS_PER_RIGHT_MATRIX, dsize_t MAX_RIGHT_MATRICES_PER_THREAD, dsize_t MAX_LEFT_MATRICES_PER_THREAD, dsize_t LEFT_ROWS_PER_ITER, typename T, typename RES>
__host__ void ccn_n_to_m_shuffle_multirow_both_multimat_both_left_rows_dispatch(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t num_left_matrices,
    dsize_t warps_per_thread_block,
    dsize_t left_rows_per_iteration
) {
    if constexpr(LEFT_ROWS_PER_ITER > 0) {
        if (LEFT_ROWS_PER_ITER == left_rows_per_iteration) {

            // Each row of cuda block corresponds to a single warp for simplified code
            constexpr dsize_t block_x_size = warp_size;

            // There will be total of num_left_matrix_groups * num_right_matrix_groups matrix_groups
            dsize_t num_left_matrix_groups = div_up(num_left_matrices, MAX_LEFT_MATRICES_PER_THREAD);
            dsize_t num_right_matrix_groups = div_up(num_right_matrices, MAX_RIGHT_MATRICES_PER_THREAD);
            // Each shift is still computed by a single thread (in the x axis), so we need as many threads
            // as there are shifts, in each matrix group
            dsize_t blocks_per_matrix_group = div_up(search_size.x, block_x_size);

            dim3 num_threads(block_x_size, warps_per_thread_block);
            dim3 num_blocks(
                // Encodes the shift in x direction and the left matrix group the thread belongs to
                blocks_per_matrix_group * num_left_matrix_groups,
                // Encodes the right matrix group the thread belongs to
                num_right_matrix_groups,
                // Encodes distribution of matrix rows between threads
                div_up(search_size.y, num_threads.y * MAX_SHIFTS_PER_RIGHT_MATRIX)
            );

            ccn_n_to_m_shuffle_multirow_both_multimat_both<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER><<<num_blocks, num_threads, 0>>>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_left_matrices,
                num_right_matrices
            );

            ccn_n_to_m_shuffle_multirow_both_multimat_both_kernel_args::record_launch(
                num_threads,
                num_blocks,
                MAX_SHIFTS_PER_RIGHT_MATRIX,
                MAX_RIGHT_MATRICES_PER_THREAD,
                MAX_LEFT_MATRICES_PER_THREAD,
                LEFT_ROWS_PER_ITER
            );
        } else {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_left_rows_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER - 1>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                left_rows_per_iteration
            );
        }
    } else {
        // TODO: Solve the -Wunused-but-set-parameter warning
        // Silence the confusing -Wunused-but-set-parameter warning
        // as we are not setting the parameters anywhere
        (void)left;
        (void)right;
        (void)out;
        (void)matrix_size;
        (void)search_size;
        (void)num_right_matrices;
        (void)num_left_matrices;
        (void)warps_per_thread_block;
        (void)left_rows_per_iteration;
        assert(false);
    }
}

template<dsize_t MAX_SHIFTS_PER_RIGHT_MATRIX, dsize_t MAX_RIGHT_MATRICES_PER_THREAD, dsize_t MAX_LEFT_MATRICES_PER_THREAD, dsize_t LEFT_ROWS_PER_ITER, typename T, typename RES>
__host__ void ccn_n_to_m_shuffle_multirow_both_multimat_both_left_mats_dispatch(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t num_left_matrices,
    dsize_t warps_per_thread_block,
    dsize_t left_matrices_per_thread,
    dsize_t left_rows_per_iteration
) {
    if constexpr(MAX_LEFT_MATRICES_PER_THREAD > 0) {
        if (MAX_LEFT_MATRICES_PER_THREAD == left_matrices_per_thread) {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_left_rows_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                left_rows_per_iteration
            );
        } else {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_left_mats_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD - 1, LEFT_ROWS_PER_ITER>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                left_matrices_per_thread,
                left_rows_per_iteration
            );
        }
    } else {
        // TODO: Solve the -Wunused-but-set-parameter warning
        // Silence the confusing -Wunused-but-set-parameter warning
        // as we are not setting the parameters anywhere
        (void)left;
        (void)right;
        (void)out;
        (void)matrix_size;
        (void)search_size;
        (void)num_right_matrices;
        (void)num_left_matrices;
        (void)warps_per_thread_block;
        (void)left_matrices_per_thread;
        (void)left_rows_per_iteration;
        assert(false);
    }
}

template<dsize_t MAX_SHIFTS_PER_RIGHT_MATRIX, dsize_t MAX_RIGHT_MATRICES_PER_THREAD, dsize_t MAX_LEFT_MATRICES_PER_THREAD, dsize_t LEFT_ROWS_PER_ITER, typename T, typename RES>
__host__ void ccn_n_to_m_shuffle_multirow_both_multimat_both_right_mats_dispatch(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t num_left_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_matrices_per_thread,
    dsize_t left_matrices_per_thread,
    dsize_t left_rows_per_iteration
) {
    if constexpr(MAX_RIGHT_MATRICES_PER_THREAD > 0) {
        if (MAX_RIGHT_MATRICES_PER_THREAD == right_matrices_per_thread) {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_left_mats_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                left_matrices_per_thread,
                left_rows_per_iteration
            );
        } else {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_right_mats_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD - 1, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                right_matrices_per_thread,
                left_matrices_per_thread,
                left_rows_per_iteration
            );
        }
    } else {
        // TODO: Solve the -Wunused-but-set-parameter warning
        // Silence the confusing -Wunused-but-set-parameter warning
        // as we are not setting the parameters anywhere
        (void)left;
        (void)right;
        (void)out;
        (void)matrix_size;
        (void)search_size;
        (void)num_right_matrices;
        (void)num_left_matrices;
        (void)warps_per_thread_block;
        (void)right_matrices_per_thread;
        (void)left_matrices_per_thread;
        (void)left_rows_per_iteration;
        assert(false);
    }
}

template<dsize_t MAX_SHIFTS_PER_RIGHT_MATRIX, dsize_t MAX_RIGHT_MATRICES_PER_THREAD, dsize_t MAX_LEFT_MATRICES_PER_THREAD, dsize_t LEFT_ROWS_PER_ITER, typename T, typename RES>
__host__ void ccn_n_to_m_shuffle_multirow_both_multimat_both_shifts_dispatch(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t num_left_matrices,
    dsize_t warps_per_thread_block,
    dsize_t shifts_per_thread_right_matrix,
    dsize_t right_matrices_per_thread,
    dsize_t left_matrices_per_thread,
    dsize_t left_rows_per_iteration
) {
    if constexpr(MAX_SHIFTS_PER_RIGHT_MATRIX > 0) {
        if (MAX_SHIFTS_PER_RIGHT_MATRIX == shifts_per_thread_right_matrix) {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_right_mats_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                right_matrices_per_thread,
                left_matrices_per_thread,
                left_rows_per_iteration
            );
        } else {
            ccn_n_to_m_shuffle_multirow_both_multimat_both_shifts_dispatch<MAX_SHIFTS_PER_RIGHT_MATRIX - 1, MAX_RIGHT_MATRICES_PER_THREAD, MAX_LEFT_MATRICES_PER_THREAD, LEFT_ROWS_PER_ITER>(
                left,
                right,
                out,
                matrix_size,
                search_size,
                num_right_matrices,
                num_left_matrices,
                warps_per_thread_block,
                shifts_per_thread_right_matrix,
                right_matrices_per_thread,
                left_matrices_per_thread,
                left_rows_per_iteration
            );
        }
    } else {
        // TODO: Solve the -Wunused-but-set-parameter warning
        // Silence the confusing -Wunused-but-set-parameter warning
        // as we are not setting the parameters anywhere
        (void)left;
        (void)right;
        (void)out;
        (void)matrix_size;
        (void)search_size;
        (void)num_right_matrices;
        (void)num_left_matrices;
        (void)warps_per_thread_block;
        (void)shifts_per_thread_right_matrix;
        (void)right_matrices_per_thread;
        (void)left_matrices_per_thread;
        (void)left_rows_per_iteration;
        assert(false);
    }
}


} // END anonymous namespace

template<typename T, typename RES>
void run_ccn_n_to_m_shuffle_multirow_both_multimat_both(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t shifts_per_thread_right_matrix,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t left_rows_per_iteration
) {
    if (warps_per_thread_block > 32) {
        throw std::runtime_error("Cuda block too large: "s + std::to_string(warps_per_thread_block) + " (max 32)");
    }

    if (shifts_per_thread_right_matrix > shifts_per_thread_per_right_matrix_limit) {
        throw std::runtime_error(
            "Too many shifts per thread in each right matrix: "s +
            std::to_string(shifts_per_thread_right_matrix) +
            "(max "s + std::to_string(shifts_per_thread_per_right_matrix_limit) +
            ")"s
        );
    }

    if (left_matrices_per_thread > left_matrices_per_thread_limit) {
        throw std::runtime_error("Too many left matrices per thread: "s +
                                 std::to_string(right_matrices_per_thread) +
                                 " (max "s +
                                 std::to_string(left_matrices_per_thread_limit) +
                                 ")"s
        );
    }

    if (right_matrices_per_thread > right_matrices_per_thread_limit) {
        throw std::runtime_error("Too many right matrices per thread: "s +
                                 std::to_string(right_matrices_per_thread) +
                                 " (max "s +
                                 std::to_string(right_matrices_per_thread_limit) +
                                 ")"s
        );
    }

    if (left_rows_per_iteration > left_rows_per_iteration_limit) {
        throw std::runtime_error(
            "Too many left rows per iteration: "s +
            std::to_string(left_rows_per_iteration) +
            "(max "s + std::to_string(left_rows_per_iteration_limit) +
            ")"s
        );
    }

    ccn_n_to_m_shuffle_multirow_both_multimat_both_shifts_dispatch<shifts_per_thread_per_right_matrix_limit, right_matrices_per_thread_limit, left_matrices_per_thread_limit, left_rows_per_iteration_limit>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        num_right_matrices,
        num_left_matrices,
        warps_per_thread_block,
        shifts_per_thread_right_matrix,
        right_matrices_per_thread,
        left_matrices_per_thread,
        left_rows_per_iteration
    );
}

// template void run_ccn_n_to_m_shuffle_multirow_both_multimat_both<int, int>(
//     const int* __restrict__ left,
//     const int* __restrict__ right,
//     int* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_left_matrices,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t shifts_per_thread_right_matrix,
//     dsize_t left_matrices_per_thread,
//     dsize_t right_matrices_per_thread,
//     dsize_t left_rows_per_iteration
// );

template void run_ccn_n_to_m_shuffle_multirow_both_multimat_both<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t shifts_per_thread_right_matrix,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t left_rows_per_iteration
);

// template void run_ccn_n_to_m_shuffle_multirow_both_multimat_both<double, double>(
//     const double* __restrict__ left,
//     const double* __restrict__ right,
//     double* __restrict__ out,
//     dsize2_t matrix_size,
//     dsize2_t search_size,
//     dsize_t num_left_matrices,
//     dsize_t num_right_matrices,
//     dsize_t warps_per_thread_block,
//     dsize_t shifts_per_thread_right_matrix,
//     dsize_t left_matrices_per_thread,
//     dsize_t right_matrices_per_thread,
//     dsize_t left_rows_per_iteration
// );

}
