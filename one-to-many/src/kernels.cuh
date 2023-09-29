#pragma once

#include "types.cuh"

namespace cross {

template<typename T>
void run_hadamard_original(
    const T* __restrict__ ref,
    T* __restrict__ deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    dsize_t num_threads
);

template<typename T>
void run_hadamard_n_to_m_over_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template<typename T>
void run_hadamard_n_to_m_over_output(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template<typename T, typename RES>
void run_cross_corr_naive_original(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    cudaStream_t cuda_stream = nullptr
);

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
);

template<typename T, typename RES>
void run_ccn_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t warps_per_thread_block
);

template<typename DIST, typename T, typename RES>
void run_ccn_shuffle_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t warps_per_thread_block,
    dsize_t max_rows_per_thread,
    cudaStream_t cuda_stream = nullptr
);

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
);

template<typename T, typename RES>
void run_ccn_warp_per_shift(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block
);

template<typename T, typename RES>
void run_ccn_warp_per_shift_simple_indexing(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block
);

template<typename DIST, typename T, typename RES>
void run_ccn_warp_per_shift_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_thread_block,
    dsize_t max_rows_per_warp
);

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
    cudaStream_t cuda_stream = nullptr
);

template<typename T, typename RES>
void run_ccn_warp_per_shift_shared_mem(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t shifts_per_cuda_block,
    dsize_t shared_mem_row_size,
    dsize_t shared_mem_rows,
    dsize_t right_matrices_per_block,
    bool strided_load,
    bool column_group_per_block,
    cudaStream_t cuda_stream = nullptr
);

template<typename DIST, typename T, typename RES>
void run_ccn_shuffle_n_to_m_multimat_both_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template<typename T, typename RES>
void run_ccn_shuffle_multirow_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t warps_per_thread_block,
    dsize_t right_rows_per_thread
);

template<typename T, typename RES>
void run_ccn_shuffle_multirow_both(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t warps_per_thread_block,
    dsize_t max_shifts_per_thread,
    dsize_t max_left_rows
);

template<typename T, typename RES>
void run_ccn_block_per_shift(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_block_size
);

template<typename T, typename RES>
void run_ccn_shuffle_multirow_right_multimat_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t right_rows_per_thread,
    dsize_t right_matrices_per_thread,
    cudaStream_t cuda_stream = nullptr
);

template<typename T, typename RES>
void run_ccn_shuffle_one_to_many_multirow_both_multimat_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t shifts_per_thread_right_matrix,
    dsize_t right_matrices_per_thread,
    dsize_t left_rows_per_iteration,
    cudaStream_t cuda_stream = nullptr
);

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
);

namespace local_mem {

template<typename DIST, typename T, typename RES>
void run_ccn_shuffle_n_to_m_multimat_both_work_distribution(
    const T *__restrict__ left,
    const T *__restrict__ right,
    RES *__restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template<typename T, typename RES>
void run_ccn_shuffle_multirow_both(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t warps_per_thread_block,
    dsize_t max_shifts_per_thread,
    dsize_t max_left_rows
);

}

}
