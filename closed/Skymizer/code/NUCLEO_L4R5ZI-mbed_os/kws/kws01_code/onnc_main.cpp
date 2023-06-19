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

static const q7_t statefulpartitionedcall_functional_1_conv2d_2_biasadd_bias_fused_bn[64] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_2_BIASADD_BIAS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_1_biasadd_bias_fused_bn[64] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_1_BIASADD_BIAS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_4_biasadd_weights_fused_bn[64 * 64 * 1 * 1] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_4_BIASADD_WEIGHTS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_3_biasadd_weights_fused_bn[64 * 64 * 1 * 1] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_3_BIASADD_WEIGHTS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_2_biasadd_weights_fused_bn[64 * 64 * 1 * 1] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_2_BIASADD_WEIGHTS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_1_biasadd_weights_fused_bn[64 * 64 * 1 * 1] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_1_BIASADD_WEIGHTS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_dense_matmul_readvariableop_0[64 * 12] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_DENSE_MATMUL_READVARIABLEOP_0;

static const q7_t statefulpartitionedcall_functional_1_conv2d_biasadd_bias_fused_bn[64] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_BIASADD_BIAS_FUSED_BN;

static const q7_t conv__101_bias_fused_bn[64] = CONV__101_BIAS_FUSED_BN;

static const q7_t conv__101_weights_fused_bn[64 * 1 * 3 * 3] = CONV__101_WEIGHTS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_biasadd_weights_fused_bn[64 * 1 * 10 * 4] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_BIASADD_WEIGHTS_FUSED_BN;

static const q7_t conv__99_bias_fused_bn[64] = CONV__99_BIAS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_dense_biasadd_readvariableop_0[12] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_DENSE_BIASADD_READVARIABLEOP_0;

static const q7_t conv__97_bias_fused_bn[64] = CONV__97_BIAS_FUSED_BN;

static const q7_t conv__95_bias_fused_bn[64] = CONV__95_BIAS_FUSED_BN;

static const q7_t conv__99_weights_fused_bn[64 * 1 * 3 * 3] = CONV__99_WEIGHTS_FUSED_BN;

static const q7_t conv__97_weights_fused_bn[64 * 1 * 3 * 3] = CONV__97_WEIGHTS_FUSED_BN;

static const q7_t conv__95_weights_fused_bn[64 * 1 * 3 * 3] = CONV__95_WEIGHTS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_4_biasadd_bias_fused_bn[64] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_4_BIASADD_BIAS_FUSED_BN;

static const q7_t statefulpartitionedcall_functional_1_conv2d_3_biasadd_bias_fused_bn[64] = STATEFULPARTITIONEDCALL_FUNCTIONAL_1_CONV2D_3_BIASADD_BIAS_FUSED_BN;

static q15_t col_buffer[1152];
static q7_t scratch_buffer[16490];
static q7_t output_data[1 * 12];

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct onnc_input_tensor_t onnc_get_input_tensor() {
const struct onnc_input_tensor_t input_tensor = { .data = (scratch_buffer + 0), .size = (1 * 49 * 10 * 1) };
return input_tensor;
}

void onnc_get_io_scaling_factors(float* in_sf, float* out_sf) {
*in_sf = 1.02776, *out_sf = 0.00787402;
}

struct onnc_output_tensor_t onnc_main() {
#if defined(RTE_Compiler_EventRecorder)
EventRecorderInitialize(EventRecordAll, 1);
#endif /* defined(RTE_Compiler_EventRecorder) */


arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/conv2d/BiasAdd__6:0 */, 10 /* dim_im_in_x */, 49 /* dim_im_in_y */, 1 /* ch_im_in */, statefulpartitionedcall_functional_1_conv2d_biasadd_weights_fused_bn /* StatefulPartitionedCall/functional_1/conv2d/BiasAdd_weights_fused_bn */, 64 /* ch_im_out */, 4 /* dim_kernel_x */, 10 /* dim_kernel_y */, 1 /* padding_x */, 4 /* padding_y */, 2 /* stride_x */, 2 /* stride_y */, statefulpartitionedcall_functional_1_conv2d_biasadd_bias_fused_bn /* StatefulPartitionedCall/functional_1/conv2d/BiasAdd_bias_fused_bn */, 1 /* bias_shift */, 5 /* out_shift */, (scratch_buffer + 490) /* StatefulPartitionedCall/functional_1/batch_normalization/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 490) /* StatefulPartitionedCall/functional_1/batch_normalization/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_depthwise_separable_conv_HWC_q7_nonsquare((scratch_buffer + 490) /* StatefulPartitionedCall/functional_1/activation/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, conv__95_weights_fused_bn /* Conv__95_weights_fused_bn */, 64 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, conv__95_bias_fused_bn /* Conv__95_bias_fused_bn */, 3 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 8490) /* StatefulPartitionedCall/functional_1/batch_normalization_1/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 8490) /* StatefulPartitionedCall/functional_1/batch_normalization_1/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 8490) /* StatefulPartitionedCall/functional_1/activation_1/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, statefulpartitionedcall_functional_1_conv2d_1_biasadd_weights_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_1/BiasAdd_weights_fused_bn */, 64 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, statefulpartitionedcall_functional_1_conv2d_1_biasadd_bias_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_1/BiasAdd_bias_fused_bn */, 6 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_2/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_2/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_depthwise_separable_conv_HWC_q7_nonsquare((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/activation_2/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, conv__97_weights_fused_bn /* Conv__97_weights_fused_bn */, 64 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, conv__97_bias_fused_bn /* Conv__97_bias_fused_bn */, 3 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/batch_normalization_3/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/batch_normalization_3/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/activation_3/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, statefulpartitionedcall_functional_1_conv2d_2_biasadd_weights_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_2/BiasAdd_weights_fused_bn */, 64 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, statefulpartitionedcall_functional_1_conv2d_2_biasadd_bias_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_2/BiasAdd_bias_fused_bn */, 5 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_4/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_4/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_depthwise_separable_conv_HWC_q7_nonsquare((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/activation_4/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, conv__99_weights_fused_bn /* Conv__99_weights_fused_bn */, 64 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, conv__99_bias_fused_bn /* Conv__99_bias_fused_bn */, 3 /* bias_shift */, 5 /* out_shift */, (scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/batch_normalization_5/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/batch_normalization_5/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/activation_5/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, statefulpartitionedcall_functional_1_conv2d_3_biasadd_weights_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_3/BiasAdd_weights_fused_bn */, 64 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, statefulpartitionedcall_functional_1_conv2d_3_biasadd_bias_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_3/BiasAdd_bias_fused_bn */, 6 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_6/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_6/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_depthwise_separable_conv_HWC_q7_nonsquare((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/activation_6/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, conv__101_weights_fused_bn /* Conv__101_weights_fused_bn */, 64 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, conv__101_bias_fused_bn /* Conv__101_bias_fused_bn */, 4 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/batch_normalization_7/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/batch_normalization_7/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/activation_7/Relu:0 */, 5 /* dim_im_in_x */, 25 /* dim_im_in_y */, 64 /* ch_im_in */, statefulpartitionedcall_functional_1_conv2d_4_biasadd_weights_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_4/BiasAdd_weights_fused_bn */, 64 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, statefulpartitionedcall_functional_1_conv2d_4_biasadd_bias_fused_bn /* StatefulPartitionedCall/functional_1/conv2d_4/BiasAdd_bias_fused_bn */, 6 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_8/FusedBatchNormV3:0 */, 5 /* dim_im_out_x */, 25 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/batch_normalization_8/FusedBatchNormV3:0 */, 1 * 64 * 25 * 5 /* size */);
{ cmsis_nn_pool_params params; params.padding.h = 0, params.padding.w = 0, params.stride.h = 25, params.stride.w = 5, params.activation.min = -128, params.activation.max = 127; cmsis_nn_dims input_shape; input_shape.n = 1, input_shape.h = 25, input_shape.w = 5, input_shape.c = 64; cmsis_nn_dims filter_shape; filter_shape.h = 25, filter_shape.w = 5; cmsis_nn_dims output_shape; output_shape.h = 1, output_shape.w = 1, output_shape.c = 64; cmsis_nn_context context; context.size = arm_avgpool_s8_get_buffer_size(input_shape.w, input_shape.c), context.buf = malloc(context.size); arm_avgpool_s8(&context, &params, &input_shape, (scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/activation_8/Relu:0 */, &filter_shape, &output_shape, (scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/average_pooling2d/AvgPool:0 */); free(context.buf); }

arm_fully_connected_q7((scratch_buffer + 8000) /* StatefulPartitionedCall/functional_1/flatten/Reshape:0 */, statefulpartitionedcall_functional_1_dense_matmul_readvariableop_0 /* StatefulPartitionedCall/functional_1/dense/MatMul/ReadVariableOp:0 */, 64 /* dim_vec */, 12 /* num_of_rows */, 0 /* bias_shift */, 7 /* out_shift */, statefulpartitionedcall_functional_1_dense_biasadd_readvariableop_0 /* StatefulPartitionedCall/functional_1/dense/BiasAdd/ReadVariableOp:0 */, (scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/dense/BiasAdd:0 */, col_buffer /* vec_buffer */);
arm_softmax_q7((scratch_buffer + 0) /* StatefulPartitionedCall/functional_1/dense/BiasAdd:0 */, 1 * 12 /* dim_vec */, (scratch_buffer + 12) /* dense */);

memcpy(output_data, (scratch_buffer + 12) /* dense */, (1 * 12));

const struct onnc_output_tensor_t output_tensor = { .data = output_data, .size = (1 * 12) };
return output_tensor;
}
#if defined(__cplusplus)
}
#endif /* defined(__cplusplus) */
