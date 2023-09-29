#pragma once

#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "stopwatch.hpp"

#include "kernels.cuh"

using namespace std::string_literals;

namespace cross {

inline bool input_sizes_divisible(dsize_t ref_num_matrices, dsize_t target_num_matrices) {
    return target_num_matrices % ref_num_matrices == 0;
}

inline bool n_to_mn_validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
    return rows > 0 && cols > 0 && left_matrices > 0 && right_matrices > 0 && input_sizes_divisible(left_matrices, right_matrices);
}

/**
 * The results should be ordered so we first have the cross-correlation of
 * the n left/reference matrices with the corresponding matrix from the first
 * n right/deformed matrices, then results of the cross-correlation of the n
 * left/reference matrices with the corresponding matrices with index [n,2*n),
 * etc. ending with the cross-correlation of the n left/reference matrices with
 * right matrices [(m-1)*n,m*n).
 *
 * When indexing, this means multiply the index of the right matrix by "m" and add left matrix index,
 * i.e. right_idx * m + left_idx
 */
template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class n_to_mn: public cross_corr_alg<T, BENCH_TYPE, ALLOC> {
public:
    n_to_mn(bool is_fft, std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :cross_corr_alg<T, BENCH_TYPE, ALLOC>(is_fft, measure_alg() ? num_measurements : 0, min_measured_time)
    {}

    static bool validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
        return rows > 0 && cols > 0 && left_matrices > 0 && right_matrices > 0 && input_sizes_divisible(left_matrices, right_matrices);
    }

protected:
    static constexpr bool measure_alg() {
        return BENCH_TYPE == BenchmarkType::Algorithm;
    }

    data_array<T> compute_valid_results() const override {
        return cpu_cross_corr_n_to_mn(this->refs(), this->targets());
    }
protected:
    static void check_input_sizes_divisible(dsize_t ref_num_matrices, dsize_t target_num_matrices) {
        if (!input_sizes_divisible(ref_num_matrices, target_num_matrices)) {
            throw std::runtime_error(
                "Invalid input data counts, "s +
                std::to_string(target_num_matrices) +
                " is not divisible by "s +
                std::to_string(ref_num_matrices)
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class cpu_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit cpu_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(false, 0, min_measured_time), refs_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};
    }

    void run_impl() override {
        cpu_cross_corr_n_to_mn(refs_, targets_, results_);
    }

private:
    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_gpu_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    naive_gpu_n_to_mn(std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(false, num_measurements, min_measured_time), refs_(), targets_(), results_()
    {}

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }
protected:
    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_targets_));
        CUCH(cudaFree(d_refs_));
    }

    void load_original(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        check_inputs();
    }

    void load_deinterleaved(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) {
        this->refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);

        dsize_t group_size = this->refs_.num_matrices();
        this->targets_ = load_interleaved_matrix_array_from_csv<T, no_padding, ALLOC>(target_path, group_size);

        check_inputs();
    }

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;
private:
    void check_inputs() const {
        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_original_alg_n_to_mn: public naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_original_alg_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>(std::size(labels), min_measured_time)
    {

    }
protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        this->load_original(ref_path, def_path);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
            run_cross_corr_naive_original(
                this->d_refs_,
                this->d_targets_,
                this->d_results_,
                this->targets_.matrix_size(),
                this->results_.matrix_size(),
                this->refs_.num_matrices(),
                this->targets_.num_matrices() / this->refs_.num_matrices()
            )
        );
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

private:
    inline static const char* labels[] = {
        "Kernel"
    };
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shuffle_multimat_right_work_distribution_n_to_mn: public naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shuffle_multimat_right_work_distribution_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>(std::size(labels), min_measured_time),
         cuda_streams_()
    {
        warps_per_thread_block_ = args.value("warps_per_thread_block", 8);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 2);
        rows_per_thread_ = args.value("rows_per_thread", 10);
        distribution_type_ = from_string(args.value("distribution_type", "rectangle"));

        auto num_cuda_streams = args.value("num_cuda_streams", 8);

        cuda_streams_.resize(num_cuda_streams);
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("warps_per_thread_block", std::to_string(warps_per_thread_block_)),
            std::make_pair("right_matrices_per_thread", std::to_string(right_matrices_per_thread_)),
            std::make_pair("rows_per_thread", std::to_string(rows_per_thread_)),
            std::make_pair("distribution_type", to_string(distribution_type_)),
            std::make_pair("num_cuda_streams", std::to_string(cuda_streams_.size()))
        };
    }
protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        this->load_deinterleaved(ref_path, def_path);
    }

    void prepare_impl() override {
        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::prepare_impl();

        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamCreate(&cuda_stream));
        }
    }

    void run_impl() override {
        // Cannot use CUDA measure as we are using multiple streams
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, true,
             switch (distribution_type_) {
                 case distribution::none:
                     start_kernels<no_distribution>();
                     break;
                 case distribution::rectangle:
                     start_kernels<rectangle_distribution>();
                     break;
                 case distribution::triangle:
                     start_kernels<triangle_distribution>();
                     break;
             }
        );
    }

    void free_impl() override {
        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamDestroy(cuda_stream));
        }

        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::free_impl();
    }

    void store_results_impl(const std::filesystem::path& out_path) const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        std::ofstream out{out_path};
        this->results().store_interleaved_to_csv(out, num_groups);
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

    data_array<T> load_valid_results(const std::filesystem::path& valid_data_path) const override {
        // Deinterleave the results
        return load_interleaved_matrix_array_from_csv<T, no_padding>(valid_data_path, this->refs().num_matrices());
    }

    data_array<T> compute_valid_results() const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        // We need to pass num_groups as group_size to again interleave the targets, it makes it basically an inverse function
        // We need to interleave them again as cpu implementation uses the original ordering
        auto deinterleaved_targets = this->targets().template interleave<std::allocator<T>>(num_groups);

        // We need to again deinterleave the results as it is compared to the deinterleaved results of our algorithm
        return cpu_cross_corr_n_to_mn(this->refs(), deinterleaved_targets).interleave(this->refs().num_matrices());
    }
private:
    inline static const char* labels[] = {
        "Kernel"
    };

    std::vector<cudaStream_t> cuda_streams_;

    dsize_t warps_per_thread_block_;
    dsize_t right_matrices_per_thread_;
    dsize_t rows_per_thread_;
    distribution distribution_type_;

    template<typename DISTRIBUTION>
    void start_kernels() {
        cuda_memset(this->d_results_, 0, this->results_.size());

        dsize_t right_mats_per_left_mat = this->targets_.num_matrices() / this->refs_.num_matrices();
        for (dsize_t ref = 0; ref < this->refs_.num_matrices(); ++ref) {
            run_ccn_shuffle_multimat_right_work_distribution<DISTRIBUTION>(
                this->d_refs_ + ref * this->refs_.matrix_size().area(),
                this->d_targets_ + (ref * right_mats_per_left_mat) * this->targets_.matrix_size().area(),
                this->d_results_ + (ref * right_mats_per_left_mat) * this->results_.matrix_size().area(),
                this->targets_.matrix_size(),
                this->results_.matrix_size(),
                right_mats_per_left_mat,
                warps_per_thread_block_,
                right_matrices_per_thread_,
                rows_per_thread_,
                cuda_streams_[ref % cuda_streams_.size()]
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shuffle_multirow_right_multimat_right_n_to_mn: public naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shuffle_multirow_right_multimat_right_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>(std::size(labels), min_measured_time),
         cuda_streams_()
    {
        warps_per_thread_block_ = args.value("warps_per_thread_block", 8);
        right_rows_per_thread_ = args.value("right_rows_per_thread", 4);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 4);

        auto num_cuda_streams = args.value("num_cuda_streams", 8);

        cuda_streams_.resize(num_cuda_streams);
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("warps_per_thread_block", std::to_string(warps_per_thread_block_)),
            std::make_pair("right_rows_per_thread", std::to_string(right_rows_per_thread_)),
            std::make_pair("right_matrices_per_thread", std::to_string(right_matrices_per_thread_)),
            std::make_pair("num_cuda_streams", std::to_string(cuda_streams_.size()))
        };
    }
protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        this->load_deinterleaved(ref_path, def_path);
    }

    void prepare_impl() override {
        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::prepare_impl();

        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamCreate(&cuda_stream));
        }
    }

    void run_impl() override {
        // Cannot use CUDA measure as we are using multiple streams
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, true,
                             start_kernels()
        );
    }

    void free_impl() override {
        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamDestroy(cuda_stream));
        }

        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::free_impl();
    }

    void store_results_impl(const std::filesystem::path& out_path) const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        std::ofstream out{out_path};
        this->results().store_interleaved_to_csv(out, num_groups);
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

    data_array<T> load_valid_results(const std::filesystem::path& valid_data_path) const override {
        // Deinterleave the results
        return load_interleaved_matrix_array_from_csv<T, no_padding>(valid_data_path, this->refs().num_matrices());
    }

    data_array<T> compute_valid_results() const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        // We need to pass num_groups as group_size to again interleave the targets, it makes it basically an inverse function
        // We need to interleave them again as cpu implementation uses the original ordering
        auto deinterleaved_targets = this->targets().template interleave<std::allocator<T>>(num_groups);

        // We need to again deinterleave the results as it is compared to the deinterleaved results of our algorithm
        return cpu_cross_corr_n_to_mn(this->refs(), deinterleaved_targets).interleave(this->refs().num_matrices());
    }
private:
    inline static const char* labels[] = {
        "Kernel"
    };

    std::vector<cudaStream_t> cuda_streams_;

    dsize_t warps_per_thread_block_;
    dsize_t right_rows_per_thread_;
    dsize_t right_matrices_per_thread_;

    void start_kernels() {
        cuda_memset(this->d_results_, 0, this->results_.size());

        dsize_t right_mats_per_left_mat = this->targets_.num_matrices() / this->refs_.num_matrices();
        for (dsize_t ref = 0; ref < this->refs_.num_matrices(); ++ref) {
            run_ccn_shuffle_multirow_right_multimat_right(
                this->d_refs_ + ref * this->refs_.matrix_size().area(),
                this->d_targets_ + (ref * right_mats_per_left_mat) * this->targets_.matrix_size().area(),
                this->d_results_ + (ref * right_mats_per_left_mat) * this->results_.matrix_size().area(),
                this->targets_.matrix_size(),
                this->results_.matrix_size(),
                right_mats_per_left_mat,
                warps_per_thread_block_,
                right_rows_per_thread_,
                right_matrices_per_thread_,
                cuda_streams_[ref % cuda_streams_.size()]
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shuffle_multirow_both_multimat_right_n_to_mn: public naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shuffle_multirow_both_multimat_right_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>(std::size(labels), min_measured_time),
         cuda_streams_()
    {
        warps_per_thread_block_ = args.value("warps_per_thread_block", 8);
        shifts_per_thread_right_matrix_ = args.value("shifts_per_thread_right_matrix", 4);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 4);
        left_rows_per_iteration_ = args.value("left_rows_per_iteration", 4);

        auto num_cuda_streams = args.value("num_cuda_streams", 8);

        cuda_streams_.resize(num_cuda_streams);
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("warps_per_thread_block", std::to_string(warps_per_thread_block_)),
            std::make_pair("shifts_per_thread_right_matrix", std::to_string(shifts_per_thread_right_matrix_)),
            std::make_pair("right_matrices_per_thread", std::to_string(right_matrices_per_thread_)),
            std::make_pair("left_rows_per_iteration", std::to_string(left_rows_per_iteration_)),
            std::make_pair("num_cuda_streams", std::to_string(cuda_streams_.size()))
        };
    }
protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        this->load_deinterleaved(ref_path, def_path);
    }

    void prepare_impl() override {
        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::prepare_impl();

        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamCreate(&cuda_stream));
        }
    }

    void run_impl() override {
        // Cannot use CUDA measure as we are using multiple streams
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, true,
            start_kernels();
        );
    }

    void free_impl() override {
        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamDestroy(cuda_stream));
        }

        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::free_impl();
    }

    void store_results_impl(const std::filesystem::path& out_path) const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        std::ofstream out{out_path};
        this->results().store_interleaved_to_csv(out, num_groups);
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

    data_array<T> load_valid_results(const std::filesystem::path& valid_data_path) const override {
        // Deinterleave the results
        return load_interleaved_matrix_array_from_csv<T, no_padding>(valid_data_path, this->refs().num_matrices());
    }

    data_array<T> compute_valid_results() const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        // We need to pass num_groups as group_size to again interleave the targets, it makes it basically an inverse function
        // We need to interleave them again as cpu implementation uses the original ordering
        auto deinterleaved_targets = this->targets().template interleave<std::allocator<T>>(num_groups);

        // We need to again deinterleave the results as it is compared to the deinterleaved results of our algorithm
        return cpu_cross_corr_n_to_mn(this->refs(), deinterleaved_targets).interleave(this->refs().num_matrices());
    }
private:
    inline static const char* labels[] = {
        "Kernel"
    };

    std::vector<cudaStream_t> cuda_streams_;

    dsize_t warps_per_thread_block_;
    dsize_t shifts_per_thread_right_matrix_;
    dsize_t right_matrices_per_thread_;
    dsize_t left_rows_per_iteration_;

    void start_kernels() {
        dsize_t right_mats_per_left_mat = this->targets_.num_matrices() / this->refs_.num_matrices();

        for (dsize_t ref = 0; ref < this->refs_.num_matrices(); ++ref) {
            run_ccn_shuffle_one_to_many_multirow_both_multimat_right(
                this->d_refs_ + ref * this->refs_.matrix_size().area(),
                this->d_targets_ + (ref * right_mats_per_left_mat) * this->targets_.matrix_size().area(),
                this->d_results_ + (ref * right_mats_per_left_mat) * this->results_.matrix_size().area(),
                this->targets_.matrix_size(),
                this->results_.matrix_size(),
                right_mats_per_left_mat,
                warps_per_thread_block_,
                shifts_per_thread_right_matrix_,
                right_matrices_per_thread_,
                left_rows_per_iteration_,
                cuda_streams_[ref % cuda_streams_.size()]
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_warp_per_shift_shared_mem_n_to_mn: public naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_warp_per_shift_shared_mem_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>(std::size(labels), min_measured_time),
         cuda_streams_()
    {
        shifts_per_thread_block_ = args.value(SHIFTS_PER_THREAD_BLOCK_ARG, 16);
        shared_mem_row_size_ = args.value(SHARED_MEM_ROW_SIZE_ARG, 32);
        shared_mem_rows_ = args.value(SHARED_MEM_ROWS_ARG, shifts_per_thread_block_);
        strided_load_ = args.value(STRIDED_LOAD_ARG, true);
        column_group_per_block_ = args.value(COLUMN_GROUP_PER_BLOCK_ARG, false);
        right_matrices_per_block_ = args.value(RIGHT_MATRICES_PER_BLOCK_ARG, 8);

        if (shared_mem_rows_ == 0) {
            shared_mem_rows_ = shifts_per_thread_block_;
        }

        // TODO: Remove this if we change the implementation to work with fewer
        //  shared mem rows than shifts per block
        if (shared_mem_rows_ < shifts_per_thread_block_) {
            throw argument_error("Invalid number of shared memory rows ["s +
                                 std::to_string(shared_mem_rows_) +
                                 "], must be greater than shifts per block [" +
                                 std::to_string(shifts_per_thread_block_) +
                                 "]",
                                 SHARED_MEM_ROWS_ARG);
        }

        auto num_cuda_streams = args.value(NUM_CUDA_STREAMS_ARG, 8);

        cuda_streams_.resize(num_cuda_streams);
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair(SHIFTS_PER_THREAD_BLOCK_ARG, std::to_string(shifts_per_thread_block_)),
            std::make_pair(SHARED_MEM_ROW_SIZE_ARG, std::to_string(shared_mem_row_size_)),
            std::make_pair(SHARED_MEM_ROWS_ARG, std::to_string(shared_mem_rows_)),
            std::make_pair(STRIDED_LOAD_ARG, std::to_string(strided_load_)),
            std::make_pair(COLUMN_GROUP_PER_BLOCK_ARG, std::to_string(column_group_per_block_)),
            std::make_pair(RIGHT_MATRICES_PER_BLOCK_ARG, std::to_string(right_matrices_per_block_)),
            std::make_pair(NUM_CUDA_STREAMS_ARG, std::to_string(cuda_streams_.size()))
        };
    }
protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        this->load_deinterleaved(ref_path, def_path);
    }

    void prepare_impl() override {
        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::prepare_impl();

        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamCreate(&cuda_stream));
        }
    }

    void run_impl() override {
        // Cannot use CUDA measure as we are using multiple streams
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, true,
                             start_kernels();
        );
    }

    void free_impl() override {
        for (auto & cuda_stream : cuda_streams_) {
            CUCH(cudaStreamDestroy(cuda_stream));
        }

        naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>::free_impl();
    }

    void store_results_impl(const std::filesystem::path& out_path) const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        std::ofstream out{out_path};
        this->results().store_interleaved_to_csv(out, num_groups);
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

    data_array<T> load_valid_results(const std::filesystem::path& valid_data_path) const override {
        // Deinterleave the results
        return load_interleaved_matrix_array_from_csv<T, no_padding>(valid_data_path, this->refs().num_matrices());
    }

    data_array<T> compute_valid_results() const override {
        dsize_t num_groups = this->targets().num_matrices() / this->refs().num_matrices();
        // We need to pass num_groups as group_size to again interleave the targets, it makes it basically an inverse function
        // We need to interleave them again as cpu implementation uses the original ordering
        auto deinterleaved_targets = this->targets().template interleave<std::allocator<T>>(num_groups);

        // We need to again deinterleave the results as it is compared to the deinterleaved results of our algorithm
        return cpu_cross_corr_n_to_mn(this->refs(), deinterleaved_targets).interleave(this->refs().num_matrices());
    }
private:
    inline static const char* labels[] = {
        "Kernel"
    };

    std::vector<cudaStream_t> cuda_streams_;

    inline static const char* SHIFTS_PER_THREAD_BLOCK_ARG = "shifts_per_thread_block";
    inline static const char* SHARED_MEM_ROW_SIZE_ARG = "shared_mem_row_size";
    inline static const char* SHARED_MEM_ROWS_ARG = "shared_mem_rows";
    inline static const char* STRIDED_LOAD_ARG = "strided_load";
    inline static const char* COLUMN_GROUP_PER_BLOCK_ARG = "column_group_per_block";
    inline static const char* RIGHT_MATRICES_PER_BLOCK_ARG = "right_matrices_per_block";
    inline static const char* NUM_CUDA_STREAMS_ARG = "num_cuda_streams";

    dsize_t shifts_per_thread_block_;
    dsize_t shared_mem_row_size_;
    dsize_t shared_mem_rows_;
    dsize_t right_matrices_per_block_;
    bool strided_load_;
    bool column_group_per_block_;

    void start_kernels() {
        dsize_t right_mats_per_left_mat = this->targets_.num_matrices() / this->refs_.num_matrices();

        for (dsize_t ref = 0; ref < this->refs_.num_matrices(); ++ref) {
            run_ccn_warp_per_shift_shared_mem(
                this->d_refs_ + ref * this->refs_.matrix_size().area(),
                this->d_targets_ + (ref * right_mats_per_left_mat) * this->targets_.matrix_size().area(),
                this->d_results_ + (ref * right_mats_per_left_mat) * this->results_.matrix_size().area(),
                this->targets_.matrix_size(),
                this->results_.matrix_size(),
                right_mats_per_left_mat,
                shifts_per_thread_block_,
                shared_mem_row_size_,
                shared_mem_rows_,
                right_matrices_per_block_,
                strided_load_,
                column_group_per_block_,
                cuda_streams_[ref % cuda_streams_.size()]
            );
        }
    }
};


template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_original_alg_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC>, public fft_alg<T, ALLOC> {
public:
    explicit fft_original_alg_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(true, std::size(labels), min_measured_time),
            refs_(), targets_(), results_(), fft_buffer_size_(0),
            fft_plan_(), fft_inv_plan_()

    {
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        results_ = data_array<T, ALLOC>{refs_.matrix_size(), targets_.num_matrices()};

        fft_buffer_size_ = refs_.matrix_size().y * (refs_.matrix_size().x / 2 + 1);

        auto num_inputs = refs_.num_matrices() + targets_.num_matrices();

        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, refs_.size() + targets_.size());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_inputs_fft_, num_inputs * fft_buffer_size_);
        );

        int input_sizes[2] = {static_cast<int>(refs_.matrix_size().y), static_cast<int>(refs_.matrix_size().x)};
        int result_sizes[2] = {static_cast<int>(results_.matrix_size().y), static_cast<int>(results_.matrix_size().x)};
        CPU_MEASURE(4, this->measure_alg(), this->sw_, false,
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, input_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs));

            FFTCH(cufftPlanMany(&fft_inv_plan_, 2, result_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
        );
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_inputs_, refs_);
        cuda_memcpy_to_device(d_inputs_ + refs_.size(), targets_);
    }

    void run_impl() override {
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, false,
            fft_real_to_complex(fft_plan_, d_inputs_, d_inputs_fft_);
        );

        CUDA_ADAPTIVE_MEASURE(1, this->measure_alg(), this->sw_,
            run_hadamard_original(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(),
                {refs_.matrix_size().y, (refs_.matrix_size().x / 2) + 1},
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices(),
                hadamard_threads_per_block_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(), d_results_)
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        FFTCH(cufftDestroy(fft_inv_plan_));
        FFTCH(cufftDestroy(fft_plan_));
        CUCH(cudaFree(d_inputs_fft_));
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_inputs_));
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    inline static const char* labels[] = {
        "Forward FFT",
        "Hadamard",
        "Inverse FFT",
        "Allocation",
        "Plan"
    };

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_inputs_;
    T* d_results_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t fft_buffer_size_;

    fft_complex_t* d_inputs_fft_;

    dsize_t hadamard_threads_per_block_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_reduced_transfer_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC>, public fft_alg<T, ALLOC> {
public:
    explicit fft_reduced_transfer_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(true, std::size(labels), min_measured_time),
            refs_(), targets_(), results_(), fft_buffer_size_(0), num_inputs_(0),
            fft_plan_(), fft_inv_plan_()
    {
        scatter_threads_per_block_  = args.value("scatter_threads_per_block", 256);
        scatter_items_per_thread_ = args.value("scatter_items_per_thread", 10);
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("scatter_threads_per_block", std::to_string(scatter_threads_per_block_)),
            std::make_pair("scatter_items_per_thread", std::to_string(scatter_items_per_thread_)),
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:

    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        padded_matrix_size_ = 2 * refs_.matrix_size();
        fft_buffer_size_ = padded_matrix_size_.y * (padded_matrix_size_.x / 2 + 1);
        // 1 for the ref matrix
        num_inputs_ = refs_.num_matrices() + targets_.num_matrices();

        // Input matrices are NOT padded
        results_ = data_array<T, ALLOC>{padded_matrix_size_, targets_.num_matrices()};

        CPU_MEASURE(5, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, refs_.size() + targets_.size());
            cuda_malloc(&d_padded_inputs_, num_inputs_ * padded_matrix_size_.area());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_padded_inputs_fft_, num_inputs_ * fft_buffer_size_);
        );

        int sizes[2] = {static_cast<int>(padded_matrix_size_.y), static_cast<int>(padded_matrix_size_.x)};
        CPU_MEASURE(6, this->measure_alg(), this->sw_, false,
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs_));

            FFTCH(cufftPlanMany(&fft_inv_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
        );
    }

    void transfer_impl() override {
        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_memcpy_to_device(d_inputs_, refs_);
            cuda_memcpy_to_device(d_inputs_ + refs_.size(), targets_);
        );

        cuda_memset(d_padded_inputs_, 0, num_inputs_ * padded_matrix_size_.area());

        CUDA_ADAPTIVE_MEASURE(4, this->measure_alg(), this->sw_,
            run_scatter(
                d_inputs_,
                d_padded_inputs_,
                refs_.matrix_size(),
                num_inputs_,
                padded_matrix_size_,
                dsize2_t{0,0},
                scatter_threads_per_block_,
                scatter_items_per_thread_
            );
        );
    }

    void run_impl() override {
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, false,
            fft_real_to_complex(fft_plan_, d_padded_inputs_, d_padded_inputs_fft_);
        );

        CUDA_ADAPTIVE_MEASURE(1, this->measure_alg(), this->sw_,
            run_hadamard_original(
                d_padded_inputs_fft_,
                d_padded_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(),
                {padded_matrix_size_.y, (padded_matrix_size_.x / 2) + 1},
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices(),
                hadamard_threads_per_block_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_padded_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(), d_results_)
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        FFTCH(cufftDestroy(fft_inv_plan_));
        FFTCH(cufftDestroy(fft_plan_));
        CUCH(cudaFree(d_padded_inputs_fft_));
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_padded_inputs_));
        CUCH(cudaFree(d_inputs_));
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    inline static const char* labels[] = {
        "Forward FFT",
        "Hadamard",
        "Inverse FFT",
        "ToDevice",
        "Scatter",
        "Allocation",
        "Plan"
    };

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_inputs_;
    T* d_padded_inputs_;
    T* d_results_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t num_inputs_;
    dsize2_t padded_matrix_size_;
    dsize_t fft_buffer_size_;

    fft_complex_t* d_padded_inputs_fft_;

    dsize_t scatter_threads_per_block_;
    dsize_t scatter_items_per_thread_;
    dsize_t hadamard_threads_per_block_;

};

}
