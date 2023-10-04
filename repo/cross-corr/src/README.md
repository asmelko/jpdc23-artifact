# Algorithms implemented by the thesis

| Algorithm and optimizations | Source file | Kernel name |
| --------------------------- | ----------- | ----------- |
| Basic | [naive_original.cu](./naive_original.cu) | `cross_corr_naive_original` |
| Warp shuffle base | [naive_shuffle.cu](./naive_shuffle.cu) | `ccn_shuffle`|
| Warp shuffle base with work distribution | [naive_shuffle.cu](./naive_shuffle.cu) | `ccn_shuffle_work_distribution` |
| Warp shuffle with multimat_right | [naive_shuffle_multimat_right.cu](./naive_shuffle_multimat_right.cu) | `ccn_shuffle_multimat_right` |
| Warp shuffle with multimat_right and work distribution | [naive_shuffle_multimat_right.cu](./naive_shuffle_multimat_right.cu) | `ccn_shuffle_multimat_right_work_distribution` |
| Warp shuffle with multimat_both and work distribution| [naive_shuffle_n_to_m_multimat_both.cu](./naive_shuffle_n_to_m_multimat_both.cu) | `ccn_shuffle_n_to_m_multimat_both_work_distribution` |
| Warp shuffle with multirow_right | [naive_shuffle_multirow_right.cu](./naive_shuffle_multirow_right.cu) | `ccn_shuffle_multirow_right` |
| Warp shuffle with multirow_both | [naive_shuffle_multirow_both.cu](./naive_shuffle_multirow_both.cu) | `ccn_shuffle_multirow_both` |
| Warp shuffle with multirow_right and multimat_right | [naive_shuffle_multirow_right_multimat_right.cu](./naive_shuffle_multirow_right_multimat_right.cu) | `ccn_shuffle_multirow_right_multimat_right` |
| Warp shuffle with multirow_both and multimat_right | [naive_shuffle_one_to_many_multirow_both_multimat_right.cu](./naive_shuffle_one_to_many_multirow_both_multimat_right.cu) | `ccn_shuffle_one_to_many_multirow_both_multimat_right` |
| Warp shuffle with multirow_both and multimat_both | [naive_shuffle_n_to_m_multirow_both_multimat_both.cu](./naive_shuffle_n_to_m_multirow_both_multimat_both.cu) | `ccn_n_to_m_shuffle_multirow_both_multimat_both` |
| Warp per shift base | [naive_warp_per_shift.cu](./naive_warp_per_shift.cu) | `ccn_warp_per_shift` |
| Warp per shift with simple indexing | [naive_warp_per_shift.cu](./naive_warp_per_shift.cu) | `ccn_warp_per_shift_simple_indexing` |
| Warp per shift with work distribution | [naive_warp_per_shift.cu](./naive_warp_per_shift.cu) | `ccn_warp_per_shift_work_distribution` |
| Warp per shift with shared memory | [naive_warp_per_shift_shared_mem.cu](./naive_warp_per_shift_shared_mem.cu) | `ccn_warp_per_shift_shared_mem` |

To use these kernels for implementation of different input types and to implement time measurements, each kernel is wrapped by one class for each input type it is used for. This class handles algorithm arguments, time measurement, data loading, allocation, transfer, kernel execution, transfer back, result store, deallocation. These classes can be found in the files [one_to_one.hpp](./one_to_one.hpp), [one_to_many.hpp](./one_to_many.hpp), [n_to_mn.hpp](./n_to_mn.hpp), and[n_to_m.hpp](./n_to_m.hpp), based on the onput type they implement.

## Algorithm compile options

Following is the full list of provided algorithm options with their default values for Release/Debug build. The structure of the option name is `<algorithm_name>_<option_name>`.
| Option name | Release value | Debug value |
|-------------|---------------|-------------|
| SHUFFLE_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT | 8 | 4 |
| SHUFFLE_MULTIROW_RIGHT_RIGHT_ROWS_LIMIT | 8 | 4 |
| SHUFFLE_MULTIROW_BOTH_SHIFTS_PER_THREAD_LIMIT | 8 | 4 |
| SHUFFLE_MULTIROW_BOTH_LEFT_ROWS_LIMIT | 4 | 2 |
| SHUFFLE_MULTIROW_BOTH_LOCAL_MEM_SHIFTS_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_MULTIROW_BOTH_LOCAL_MEM_LEFT_ROWS_LIMIT | 4 | 2 |
| SHUFFLE_MULTIROW_RIGHT_MULTIMAT_RIGHT_RIGHT_ROWS_LIMIT | 4 | 2 |
| SHUFFLE_MULTIROW_RIGHT_MULTIMAT_RIGHT_RIGHT_MATS_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIMAT_BOTH_LOCAL_MEM_LEFT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIMAT_BOTH_LOCAL_MEM_RIGHT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_SHIFTS_PER_THREAD_PER_RIGHT_MATRIX_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_ROWS_PER_ITERATION_LIMIT | 4 | 2 |
| SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_SHIFTS_PER_RIGHT_MATRIX_LIMIT | 4 | 2 |
| SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT | 4 | 2 |
| SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_LEFT_ROWS_PER_ITERATION_LIMIT | 4 | 2 |
| WARP_PER_SHIFT_SHARED_MEM_RIGHT_MATRICES_PER_BLOCK_LIMIT | 8 | 2 |

The default build is mainly limited by the `SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH` and `SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT`. Reducing values for options of these two algorithms will improve the compilation time. The number of generated functions is approximately equal to the values of the options for the given algorithm multiplied together. As these two algorithms both have 4 options, they generate the most functions and take the longest time to compile.