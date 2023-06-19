#include "arm_math_types.h"
#include "arm_nnfunctions.h"
#include "onnc_weight.h"
#include "onnc_main.h"
#include "onnc_runtime.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_RTE_)
#include "RTE_Components.h"

#if defined(RTE_Compiler_EventRecorder)
#include "EventRecorder.h"

#endif /* defined(RTE_Compiler_EventRecorder) */
#endif /* defined(_RTE_) */

static const q7_t functional_1_dense_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_9_biasadd_readvariableop_resource[640] = FUNCTIONAL_1_DENSE_9_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_8_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_8_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_7_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_7_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_6_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_6_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_5_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_5_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_4_biasadd_readvariableop_resource[8] = FUNCTIONAL_1_DENSE_4_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_3_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_3_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_2_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_2_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t functional_1_dense_1_biasadd_readvariableop_resource[128] = FUNCTIONAL_1_DENSE_1_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t const_fold_opt__53[128 * 128] = CONST_FOLD_OPT__53;

static const q7_t const_fold_opt__59[640 * 128] = CONST_FOLD_OPT__59;

static const q7_t const_fold_opt__58[8 * 128] = CONST_FOLD_OPT__58;

static const q7_t const_fold_opt__57[128 * 8] = CONST_FOLD_OPT__57;

static const q7_t const_fold_opt__56[128 * 128] = CONST_FOLD_OPT__56;

static const q7_t const_fold_opt__55[128 * 128] = CONST_FOLD_OPT__55;

static const q7_t const_fold_opt__54[128 * 640] = CONST_FOLD_OPT__54;

static const q7_t const_fold_opt__62[128 * 128] = CONST_FOLD_OPT__62;

static const q7_t const_fold_opt__61[128 * 128] = CONST_FOLD_OPT__61;

static const q7_t const_fold_opt__60[128 * 128] = CONST_FOLD_OPT__60;

static q15_t col_buffer[640];
static q7_t scratch_buffer[896];
static q7_t output_data[1 * 640];

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct onnc_input_tensor_t onnc_get_input_tensor() {
const struct onnc_input_tensor_t input_tensor = { .data = (scratch_buffer + 0), .size = (1 * 640) };
return input_tensor;
}

void onnc_get_io_scaling_factors(float* in_sf, float* out_sf) {
*in_sf = 1.49768, *out_sf = 1.02352;
}

struct onnc_output_tensor_t onnc_main() {
#if defined(RTE_Compiler_EventRecorder)
EventRecorderInitialize(EventRecordAll, 1);
#endif /* defined(RTE_Compiler_EventRecorder) */

arm_fully_connected_q7((scratch_buffer + 0) /* input_1 */, const_fold_opt__59 /* const_fold_opt__59 */, 640 /* dim_vec */, 128 /* num_of_rows */, 10 /* bias_shift */, 8 /* out_shift */, functional_1_dense_biasadd_readvariableop_resource /* functional_1/dense/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 640) /* Add__8:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 640) /* Add__8:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 640) /* Relu__5:0 */, const_fold_opt__53 /* const_fold_opt__53 */, 128 /* dim_vec */, 128 /* num_of_rows */, 5 /* bias_shift */, 6 /* out_shift */, functional_1_dense_1_biasadd_readvariableop_resource /* functional_1/dense_1/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 0) /* Add__13:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 0) /* Add__13:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 0) /* Relu__10:0 */, const_fold_opt__60 /* const_fold_opt__60 */, 128 /* dim_vec */, 128 /* num_of_rows */, 3 /* bias_shift */, 5 /* out_shift */, functional_1_dense_2_biasadd_readvariableop_resource /* functional_1/dense_2/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 128) /* Add__18:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 128) /* Add__18:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 128) /* Relu__15:0 */, const_fold_opt__62 /* const_fold_opt__62 */, 128 /* dim_vec */, 128 /* num_of_rows */, 2 /* bias_shift */, 3 /* out_shift */, functional_1_dense_3_biasadd_readvariableop_resource /* functional_1/dense_3/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 0) /* Add__23:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 0) /* Add__23:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 0) /* Relu__20:0 */, const_fold_opt__57 /* const_fold_opt__57 */, 128 /* dim_vec */, 8 /* num_of_rows */, 6 /* bias_shift */, 7 /* out_shift */, functional_1_dense_4_biasadd_readvariableop_resource /* functional_1/dense_4/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 128) /* Add__28:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 128) /* Add__28:0 */, 1 * 8 /* size */);
arm_fully_connected_q7((scratch_buffer + 128) /* Relu__25:0 */, const_fold_opt__58 /* const_fold_opt__58 */, 8 /* dim_vec */, 128 /* num_of_rows */, 6 /* bias_shift */, 5 /* out_shift */, functional_1_dense_5_biasadd_readvariableop_resource /* functional_1/dense_5/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 0) /* Add__33:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 0) /* Add__33:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 0) /* Relu__30:0 */, const_fold_opt__56 /* const_fold_opt__56 */, 128 /* dim_vec */, 128 /* num_of_rows */, 5 /* bias_shift */, 6 /* out_shift */, functional_1_dense_6_biasadd_readvariableop_resource /* functional_1/dense_6/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 128) /* Add__38:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 128) /* Add__38:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 128) /* Relu__35:0 */, const_fold_opt__61 /* const_fold_opt__61 */, 128 /* dim_vec */, 128 /* num_of_rows */, 5 /* bias_shift */, 6 /* out_shift */, functional_1_dense_7_biasadd_readvariableop_resource /* functional_1/dense_7/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 0) /* Add__43:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 0) /* Add__43:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 0) /* Relu__40:0 */, const_fold_opt__55 /* const_fold_opt__55 */, 128 /* dim_vec */, 128 /* num_of_rows */, 6 /* bias_shift */, 7 /* out_shift */, functional_1_dense_8_biasadd_readvariableop_resource /* functional_1/dense_8/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 128) /* Add__48:0 */, col_buffer /* vec_buffer */);
arm_relu_q7((scratch_buffer + 128) /* Add__48:0 */, 1 * 128 /* size */);
arm_fully_connected_q7((scratch_buffer + 128) /* Relu__45:0 */, const_fold_opt__54 /* const_fold_opt__54 */, 128 /* dim_vec */, 640 /* num_of_rows */, 5 /* bias_shift */, 10 /* out_shift */, functional_1_dense_9_biasadd_readvariableop_resource /* functional_1/dense_9/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 256) /* Identity */, col_buffer /* vec_buffer */);

memcpy(output_data, (scratch_buffer + 256) /* Identity */, (1 * 640));

const struct onnc_output_tensor_t output_tensor = { .data = output_data, .size = (1 * 640) };
return output_tensor;
}
#if defined(__cplusplus)
}
#endif /* defined(__cplusplus) */
