#pragma once

#include <vector>
#include <chrono>
#include <filesystem>

#include <cufft.h>

#include "matrix.hpp"
#include "stopwatch.hpp"

#include "kernels.cuh"

#include "fft_helpers.hpp"
#include "run_args.hpp"

namespace cross {

using sw_clock = std::chrono::high_resolution_clock;

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class cross_corr_alg {
public:
    using data_type = T;
    using allocator = ALLOC;
    constexpr static BenchmarkType benchmarking_type = BENCH_TYPE;

    cross_corr_alg(bool is_fft, std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :is_fft_(is_fft), sw_(measure_common() ? std::size(labels) : num_measurements, min_measured_time)
    {}

    void load(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) {
        CPU_MEASURE(0, measure_common(), this->sw_, false,
            load_impl(ref_path, target_path);
        );
    }

    void prepare() {
        CPU_MEASURE(1, measure_common(), this->sw_, false,
            prepare_impl();
        );
    }

    void transfer() {
        CPU_MEASURE(2, measure_common(), this->sw_, true,
            transfer_impl();
        );
    }

    void run() {
        CPU_ADAPTIVE_MEASURE(3, measure_common(), this->sw_, true,
            run_impl();
        );
    }

    void finalize() {
        CPU_MEASURE(4, measure_common(), this->sw_, true,
            finalize_impl();
        );
    }

    void free() {
        CPU_MEASURE(5, measure_common(), this->sw_, false,
            free_impl();
        );
    }

    void store_results(const std::filesystem::path& out_path) {
        CPU_MEASURE(6, measure_common(), this->sw_, false,
            store_results_impl(out_path);
        );
    }

    void collect_measurements() {
        sw_.cuda_collect();
    }

    void reset_measurements() {
        sw_.reset();
    }

    virtual const data_array<T, ALLOC>& refs() const = 0;

    virtual const data_array<T, ALLOC>& targets() const = 0;

    virtual const data_array<T, ALLOC>& results() const = 0;



    [[nodiscard]] validation_results validate(const std::optional<std::filesystem::path>& valid_data_path = std::nullopt) const {
        auto valid = get_valid_results(valid_data_path);
        if (this->is_fft()) {
            return validate_result(normalize_fft_results(this->results()), valid);
        } else {
            return validate_result(this->results(), valid);
        }
    }

    [[nodiscard]] std::vector<const char*> measurement_labels() const {
        return measure_common() ?
            std::vector<const char*>(std::begin(labels), std::end(labels)) :
            measurement_labels_impl();
    };

    [[nodiscard]] bool is_fft() const {
        return is_fft_;
    }

    [[nodiscard]] const std::vector<stopwatch<sw_clock>::result>& measurements() const {
        return sw_.results();
    }

    [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> additional_properties() const {
        return std::vector<std::pair<std::string, std::string>>{};
    }
protected:

    bool is_fft_;
    stopwatch<sw_clock> sw_;

    static void check_matrices_same_size(const data_array<T, ALLOC>& ref, const data_array<T, ALLOC>& target) {
        if (ref.matrix_size() != target.matrix_size()) {
            throw std::runtime_error(
                "Invalid input matrix sizes, expected ref and target to be the same size: ref = "s +
                to_string(ref.matrix_size()) +
                " target = "s +
                to_string(target.matrix_size())
            );
        }
    }

    virtual void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) = 0;

    virtual void prepare_impl() {

    }

    virtual void transfer_impl() {

    }

    virtual void run_impl() = 0;

    virtual void finalize_impl() {

    }

    virtual void free_impl() {

    }

    virtual void store_results_impl(const std::filesystem::path& out_path) const {
        std::ofstream out{out_path};
        results().store_to_csv(out);
    }

    virtual data_array<T> load_valid_results(const std::filesystem::path& valid_data_path) const {
        return load_matrix_array_from_csv<T, no_padding>(valid_data_path);
    }

    virtual data_array<T> compute_valid_results() const = 0;

    [[nodiscard]] virtual std::vector<const char*> measurement_labels_impl() const {
        return std::vector<const char*>{};
    }

private:
    inline static const char* labels[] = {
        "Load",
        "Prepare",
        "Transfer",
        "Run",
        "Finalize",
        "Free",
        "Store"
    };

    static constexpr bool measure_common() {
        return BENCH_TYPE == BenchmarkType::CommonSteps;
    }

    data_array<T> get_valid_results(const std::optional<std::filesystem::path>& valid_data_path = std::nullopt) const {
        if (valid_data_path.has_value()) {
            return load_valid_results(valid_data_path.value());
        } else {
            return compute_valid_results();
        }
    }
};

}
