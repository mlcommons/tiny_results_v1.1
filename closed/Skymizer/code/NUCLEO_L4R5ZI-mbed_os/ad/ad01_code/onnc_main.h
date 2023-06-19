#pragma once

#include "arm_math_types.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct onnc_input_tensor_t { int8_t* data; size_t size; };
struct onnc_output_tensor_t { int8_t* data; size_t size; };

struct onnc_input_tensor_t onnc_get_input_tensor();
void onnc_get_io_scaling_factors(float* in_sf, float* out_sf);
struct onnc_output_tensor_t onnc_main();
#if defined(__cplusplus)
}
#endif /* defined(__cplusplus) */
