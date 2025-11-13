#include <tensor_defs.h>

namespace tensor {

/**
 * Get the name of a backend as a string
 */
std::string toString(Backend backend) {
    switch (backend) {
        case Backend::CPU: return "CPU";
        case Backend::BLAS: return "BLAS";
        case Backend::GPU: return "GPU";
        default: return "Unknown";
    }
}

} // namespace tensor

