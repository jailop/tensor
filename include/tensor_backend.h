#ifndef _TENSOR_BACKEND_H
#define _TENSOR_BACKEND_H

#include <tensor_defs.h>

namespace tensor {

/**
 * This function checks at runtime which backend is available and will be used
 * for new tensor operations. Priority: GPU > BLAS > CPU
 */
Backend get_active_backend();

} // namespace tensor

#endif // _TENSOR_BACKEND_H
