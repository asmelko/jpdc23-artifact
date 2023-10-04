#pragma once

#include <iostream>
#include "fft_helpers.hpp"

namespace cross {

template<typename T, typename ALLOC>
class fft_alg {
public:
    virtual const data_array<T, ALLOC>& results() const = 0;

    virtual void store_results(const std::filesystem::path& out_path, bool normalize) {
        std::ofstream out{out_path};
        if (normalize) {
            normalize_fft_results(results()).store_to_csv(out);
        } else {
            results().store_to_csv(out);
        }
    }
};

}
