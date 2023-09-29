#pragma once

#include <iostream>
#include <string>
#include <optional>

#include "validate.hpp"
#include "csv.hpp"

/**
 * Toggleable ostream which serves as a decorator for another ostream.
 * Can be toggled on or off, ignoring any message print requests if toggled off
 */
namespace cross {

class simple_logger {
public:
    simple_logger(bool print_progress, bool append_result_stats)
        :print_progress_(print_progress), append_result_stats_(append_result_stats)
    {

    }

    void log(const std::string& message) const {
        if (print_progress_) {
            std::cout << message << std::endl;
        }
    }

    void result_stats(const validation_results& results, std::optional<bool> print_header_override = std::nullopt) const {
        if (print_progress_) {
            std::cout << results;
        } else {
            if ((print_header_override.has_value() && *print_header_override) ||
                (!print_header_override.has_value() && !append_result_stats_)
            ) {
                std::cout << results.csv_header() << "\n";
            }
            std::cout << results.csv_data() << "\n";
        }
    }
private:
    bool print_progress_;
    bool append_result_stats_;
};

}
