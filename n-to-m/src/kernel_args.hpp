#include <vector_types.h>
#include <string>
#include <unordered_map>

#include "types.cuh"

namespace cross {

class kernel_args {
public:
    dim3 block_size_;
    dim3 grid_size_;
    dsize_t shared_mem_bytes_;

    [[nodiscard]] virtual std::unordered_map<std::string, std::string> get_additional_args() const {
        return std::unordered_map<std::string, std::string>{};
    };

    [[nodiscard]] std::string string() const;
protected:
    kernel_args();

    void set_common(
        dim3 block_size,
        dim3 grid_size,
        dsize_t shared_mem_bytes
    );
};

void set_last_kernel_launch_args(const kernel_args* args);
const kernel_args* get_last_kernel_launch_args();

std::string last_kernel_launch_args_string();

}
