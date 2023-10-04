#include "kernel_args.hpp"

#include <sstream>

namespace cross {

static const kernel_args* last_kernel_launch_args = nullptr;

const kernel_args* get_last_kernel_launch_args() {
    return last_kernel_launch_args;
}

void set_last_kernel_launch_args(const kernel_args* args) {
    last_kernel_launch_args = args;
}

std::string last_kernel_launch_args_string() {
    const kernel_args* latest_args = get_last_kernel_launch_args();
    if (latest_args) {
        return latest_args->string();
    } else {
        return "No kernel launch";
    }
}

kernel_args::kernel_args()
    : block_size_(), grid_size_(), shared_mem_bytes_(0)
{

}

std::string kernel_args::string() const {
    std::stringstream out;
    out << "Block size: [" << block_size_.x << ", " << block_size_.y << ", " << block_size_.z << "]" <<
        "\nGrid size: [" << grid_size_.x << ", " << grid_size_.y << ", " << grid_size_.z << "]" <<
        "\nShared memory: " << shared_mem_bytes_ << "B\n";

    for (auto const& [key, val]: get_additional_args()) {
        out << key << ": " << val << "\n";
    }

    return out.str();
}

void kernel_args::set_common(
    dim3 block_size,
    dim3 grid_size,
    dsize_t shared_mem_bytes
) {
    this->block_size_ = block_size;
    this->grid_size_ = grid_size;
    this->shared_mem_bytes_ = shared_mem_bytes;
}

}
