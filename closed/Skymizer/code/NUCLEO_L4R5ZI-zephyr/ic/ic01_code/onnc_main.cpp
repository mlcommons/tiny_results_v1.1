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

static const q7_t model_activation_3_relu_model_batch_normalization_3_fusedbatchnormv3_model_conv2d_3_biasadd_readvariableop_resource_model_conv2d_3_biasadd_model_conv2d_5_conv2d_model_conv2d_3_conv2d[32] = MODEL_ACTIVATION_3_RELU_MODEL_BATCH_NORMALIZATION_3_FUSEDBATCHNORMV3_MODEL_CONV2D_3_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_3_BIASADD_MODEL_CONV2D_5_CONV2D_MODEL_CONV2D_3_CONV2D;

static const q7_t const_fold_opt__116[16 * 16 * 3 * 3] = CONST_FOLD_OPT__116;

static const q7_t const_fold_opt__118[16 * 3 * 3 * 3] = CONST_FOLD_OPT__118;

static const q7_t model_batch_normalization_6_fusedbatchnormv3_model_conv2d_7_biasadd_readvariableop_resource_model_conv2d_7_biasadd[64] = MODEL_BATCH_NORMALIZATION_6_FUSEDBATCHNORMV3_MODEL_CONV2D_7_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_7_BIASADD;

static const q7_t const_fold_opt__120[64 * 10] = CONST_FOLD_OPT__120;

static const q7_t const_fold_opt__123[64 * 64 * 3 * 3] = CONST_FOLD_OPT__123;

static const q7_t const_fold_opt__119[32 * 16 * 1 * 1] = CONST_FOLD_OPT__119;

static const q7_t const_fold_opt__127[64 * 32 * 1 * 1] = CONST_FOLD_OPT__127;

static const q7_t const_fold_opt__126[16 * 16 * 3 * 3] = CONST_FOLD_OPT__126;

static const q7_t const_fold_opt__124[32 * 32 * 3 * 3] = CONST_FOLD_OPT__124;

static const q7_t model_activation_1_relu_model_batch_normalization_1_fusedbatchnormv3_model_conv2d_1_biasadd_readvariableop_resource_model_conv2d_1_biasadd_model_conv2d_2_conv2d_model_conv2d_1_conv2d[16] = MODEL_ACTIVATION_1_RELU_MODEL_BATCH_NORMALIZATION_1_FUSEDBATCHNORMV3_MODEL_CONV2D_1_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_1_BIASADD_MODEL_CONV2D_2_CONV2D_MODEL_CONV2D_1_CONV2D;

static const q7_t model_batch_normalization_4_fusedbatchnormv3_model_conv2d_4_biasadd_readvariableop_resource_model_conv2d_4_biasadd[32] = MODEL_BATCH_NORMALIZATION_4_FUSEDBATCHNORMV3_MODEL_CONV2D_4_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_4_BIASADD;

static const q7_t const_fold_opt__121[64 * 32 * 3 * 3] = CONST_FOLD_OPT__121;

static const q7_t model_conv2d_8_biasadd_model_conv2d_8_conv2d_model_conv2d_8_biasadd_readvariableop_resource[64] = MODEL_CONV2D_8_BIASADD_MODEL_CONV2D_8_CONV2D_MODEL_CONV2D_8_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t model_activation_5_relu_model_batch_normalization_5_fusedbatchnormv3_model_conv2d_6_biasadd_readvariableop_resource_model_conv2d_6_biasadd_model_conv2d_8_conv2d_model_conv2d_6_conv2d[64] = MODEL_ACTIVATION_5_RELU_MODEL_BATCH_NORMALIZATION_5_FUSEDBATCHNORMV3_MODEL_CONV2D_6_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_6_BIASADD_MODEL_CONV2D_8_CONV2D_MODEL_CONV2D_6_CONV2D;

static const q7_t model_dense_biasadd_readvariableop_resource[10] = MODEL_DENSE_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t model_batch_normalization_2_fusedbatchnormv3_model_conv2d_2_biasadd_readvariableop_resource_model_conv2d_2_biasadd[16] = MODEL_BATCH_NORMALIZATION_2_FUSEDBATCHNORMV3_MODEL_CONV2D_2_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_2_BIASADD;

static const q7_t model_conv2d_5_biasadd_model_conv2d_5_conv2d_model_conv2d_5_biasadd_readvariableop_resource[32] = MODEL_CONV2D_5_BIASADD_MODEL_CONV2D_5_CONV2D_MODEL_CONV2D_5_BIASADD_READVARIABLEOP_RESOURCE;

static const q7_t const_fold_opt__112[32 * 16 * 3 * 3] = CONST_FOLD_OPT__112;

static const q7_t model_activation_relu_model_batch_normalization_fusedbatchnormv3_model_conv2d_biasadd_readvariableop_resource_model_conv2d_biasadd_model_conv2d_2_conv2d_model_conv2d_conv2d[16] = MODEL_ACTIVATION_RELU_MODEL_BATCH_NORMALIZATION_FUSEDBATCHNORMV3_MODEL_CONV2D_BIASADD_READVARIABLEOP_RESOURCE_MODEL_CONV2D_BIASADD_MODEL_CONV2D_2_CONV2D_MODEL_CONV2D_CONV2D;

static const int mean_data[3] = MEAN_DATA;

static const unsigned scale_data[3] = SCALE_DATA;

static q15_t col_buffer[1152];
static q7_t scratch_buffer[52224];
static q7_t output_data[1 * 10];

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct onnc_input_tensor_t onnc_get_input_tensor() {
const struct onnc_input_tensor_t input_tensor = { .data = (uint8_t*)((scratch_buffer + 0)), .size = (1 * 3 * 32 * 32) };
return input_tensor;
}

struct onnc_output_tensor_t onnc_main() {
#if defined(RTE_Compiler_EventRecorder)
EventRecorderInitialize(EventRecordAll, 1);
#endif /* defined(RTE_Compiler_EventRecorder) */

{
const struct onnc_input_tensor_t input_buffer = onnc_get_input_tensor();
for (size_t pixel = 0; pixel < 1 * 3 * 32 * 32; pixel += 3) {
*((scratch_buffer + 0) + pixel + 0) = (q7_t)(__SSAT(((int)(input_buffer.data[pixel + 0]) - mean_data[0]) >> (scale_data[0]), 8));
*((scratch_buffer + 0) + pixel + 1) = (q7_t)(__SSAT(((int)(input_buffer.data[pixel + 1]) - mean_data[1]) >> (scale_data[1]), 8));
*((scratch_buffer + 0) + pixel + 2) = (q7_t)(__SSAT(((int)(input_buffer.data[pixel + 2]) - mean_data[2]) >> (scale_data[2]), 8));
}
}

arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 0) /* input_1 */, 32 /* dim_im_in_x */, 32 /* dim_im_in_y */, 3 /* ch_im_in */, const_fold_opt__118 /* const_fold_opt__118 */, 16 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, model_activation_relu_model_batch_normalization_fusedbatchnormv3_model_conv2d_biasadd_readvariableop_resource_model_conv2d_biasadd_model_conv2d_2_conv2d_model_conv2d_conv2d /* model/activation/Relu;model/batch_normalization/FusedBatchNormV3;model/conv2d/BiasAdd/ReadVariableOp/resource;model/conv2d/BiasAdd;model/conv2d_2/Conv2D;model/conv2d/Conv2D */, 6 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 3072) /* model/activation/Relu;model/batch_normalization/FusedBatchNormV3;model/conv2d/BiasAdd/ReadVariableOp/resource;model/conv2d/BiasAdd;model/conv2d_2/Conv2D;model/conv2d/Conv2D1 */, 32 /* dim_im_out_x */, 32 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 3072) /* model/activation/Relu;model/batch_normalization/FusedBatchNormV3;model/conv2d/BiasAdd/ReadVariableOp/resource;model/conv2d/BiasAdd;model/conv2d_2/Conv2D;model/conv2d/Conv2D1 */, 1 * 16 * 32 * 32 /* size */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 3072) /* Relu__5:0 */, 32 /* dim_im_in_x */, 32 /* dim_im_in_y */, 16 /* ch_im_in */, const_fold_opt__126 /* const_fold_opt__126 */, 16 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, model_activation_1_relu_model_batch_normalization_1_fusedbatchnormv3_model_conv2d_1_biasadd_readvariableop_resource_model_conv2d_1_biasadd_model_conv2d_2_conv2d_model_conv2d_1_conv2d /* model/activation_1/Relu;model/batch_normalization_1/FusedBatchNormV3;model/conv2d_1/BiasAdd/ReadVariableOp/resource;model/conv2d_1/BiasAdd;model/conv2d_2/Conv2D;model/conv2d_1/Conv2D */, 7 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 19456) /* model/activation_1/Relu;model/batch_normalization_1/FusedBatchNormV3;model/conv2d_1/BiasAdd/ReadVariableOp/resource;model/conv2d_1/BiasAdd;model/conv2d_2/Conv2D;model/conv2d_1/Conv2D1 */, 32 /* dim_im_out_x */, 32 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 19456) /* model/activation_1/Relu;model/batch_normalization_1/FusedBatchNormV3;model/conv2d_1/BiasAdd/ReadVariableOp/resource;model/conv2d_1/BiasAdd;model/conv2d_2/Conv2D;model/conv2d_1/Conv2D1 */, 1 * 16 * 32 * 32 /* size */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 19456) /* Relu__8:0 */, 32 /* dim_im_in_x */, 32 /* dim_im_in_y */, 16 /* ch_im_in */, const_fold_opt__116 /* const_fold_opt__116 */, 16 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, model_batch_normalization_2_fusedbatchnormv3_model_conv2d_2_biasadd_readvariableop_resource_model_conv2d_2_biasadd /* model/batch_normalization_2/FusedBatchNormV3;model/conv2d_2/BiasAdd/ReadVariableOp/resource;model/conv2d_2/BiasAdd */, 5 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 35840) /* model/batch_normalization_2/FusedBatchNormV3;model/conv2d_2/BiasAdd/ReadVariableOp/resource;model/conv2d_2/BiasAdd;model/conv2d_2/Conv2D */, 32 /* dim_im_out_x */, 32 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_elementwise_add_s8((scratch_buffer + 3072) /* Relu__5:0 */, (scratch_buffer + 35840) /* model/batch_normalization_2/FusedBatchNormV3;model/conv2d_2/BiasAdd/ReadVariableOp/resource;model/conv2d_2/BiasAdd;model/conv2d_2/Conv2D */, 0 /* input_1_offset */, 2147483647 /* input_1_mult */, -2 /* input_1_shift */, 0 /* input_2_offset */, 1110985728 /* input_2_mult */, 0 /* input_2_shift */, 2 /* left_shift */, (scratch_buffer + 3072) /* model/activation_2/Relu;model/add/add */, 0 /* out_offset */, 1782413696 /* out_mult */, 0 /* out_shift */, -128 /* out_activation_min */, 127 /* out_activation_max */, (1 * 16 * 32 * 32) /* block_size */);
arm_relu_q7((scratch_buffer + 3072) /* model/activation_2/Relu;model/add/add */, 1 * 16 * 32 * 32 /* size */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 3072) /* Relu__12:0 */, 32 /* dim_im_in_x */, 32 /* dim_im_in_y */, 16 /* ch_im_in */, const_fold_opt__112 /* const_fold_opt__112 */, 32 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 2 /* stride_x */, 2 /* stride_y */, model_activation_3_relu_model_batch_normalization_3_fusedbatchnormv3_model_conv2d_3_biasadd_readvariableop_resource_model_conv2d_3_biasadd_model_conv2d_5_conv2d_model_conv2d_3_conv2d /* model/activation_3/Relu;model/batch_normalization_3/FusedBatchNormV3;model/conv2d_3/BiasAdd/ReadVariableOp/resource;model/conv2d_3/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_3/Conv2D */, 7 /* bias_shift */, 9 /* out_shift */, (scratch_buffer + 27648) /* model/activation_3/Relu;model/batch_normalization_3/FusedBatchNormV3;model/conv2d_3/BiasAdd/ReadVariableOp/resource;model/conv2d_3/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_3/Conv2D1 */, 16 /* dim_im_out_x */, 16 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 27648) /* model/activation_3/Relu;model/batch_normalization_3/FusedBatchNormV3;model/conv2d_3/BiasAdd/ReadVariableOp/resource;model/conv2d_3/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_3/Conv2D1 */, 1 * 32 * 16 * 16 /* size */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 27648) /* Relu__14:0 */, 16 /* dim_im_in_x */, 16 /* dim_im_in_y */, 32 /* ch_im_in */, const_fold_opt__124 /* const_fold_opt__124 */, 32 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, model_batch_normalization_4_fusedbatchnormv3_model_conv2d_4_biasadd_readvariableop_resource_model_conv2d_4_biasadd /* model/batch_normalization_4/FusedBatchNormV3;model/conv2d_4/BiasAdd/ReadVariableOp/resource;model/conv2d_4/BiasAdd */, 6 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 35840) /* model/batch_normalization_4/FusedBatchNormV3;model/conv2d_4/BiasAdd/ReadVariableOp/resource;model/conv2d_4/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_4/Conv2D */, 16 /* dim_im_out_x */, 16 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 3072) /* Relu__12:0 */, 32 /* dim_im_in_x */, 32 /* dim_im_in_y */, 16 /* ch_im_in */, const_fold_opt__119 /* const_fold_opt__119 */, 32 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 2 /* stride_x */, 2 /* stride_y */, model_conv2d_5_biasadd_model_conv2d_5_conv2d_model_conv2d_5_biasadd_readvariableop_resource /* model/conv2d_5/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_5/BiasAdd/ReadVariableOp/resource */, 4 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 19456) /* model/conv2d_5/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_5/BiasAdd/ReadVariableOp/resource1 */, 16 /* dim_im_out_x */, 16 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_elementwise_add_s8((scratch_buffer + 19456) /* model/conv2d_5/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_5/BiasAdd/ReadVariableOp/resource1 */, (scratch_buffer + 35840) /* model/batch_normalization_4/FusedBatchNormV3;model/conv2d_4/BiasAdd/ReadVariableOp/resource;model/conv2d_4/BiasAdd;model/conv2d_5/Conv2D;model/conv2d_4/Conv2D */, 0 /* input_1_offset */, 2147483647 /* input_1_mult */, -2 /* input_1_shift */, 0 /* input_2_offset */, 1207102336 /* input_2_mult */, 0 /* input_2_shift */, 2 /* left_shift */, (scratch_buffer + 19456) /* model/activation_4/Relu;model/add_1/add */, 0 /* out_offset */, 1870703744 /* out_mult */, 0 /* out_shift */, -128 /* out_activation_min */, 127 /* out_activation_max */, (1 * 32 * 16 * 16) /* block_size */);
arm_relu_q7((scratch_buffer + 19456) /* model/activation_4/Relu;model/add_1/add */, 1 * 32 * 16 * 16 /* size */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 19456) /* Relu__19:0 */, 16 /* dim_im_in_x */, 16 /* dim_im_in_y */, 32 /* ch_im_in */, const_fold_opt__121 /* const_fold_opt__121 */, 64 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 2 /* stride_x */, 2 /* stride_y */, model_activation_5_relu_model_batch_normalization_5_fusedbatchnormv3_model_conv2d_6_biasadd_readvariableop_resource_model_conv2d_6_biasadd_model_conv2d_8_conv2d_model_conv2d_6_conv2d /* model/activation_5/Relu;model/batch_normalization_5/FusedBatchNormV3;model/conv2d_6/BiasAdd/ReadVariableOp/resource;model/conv2d_6/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_6/Conv2D */, 8 /* bias_shift */, 9 /* out_shift */, (scratch_buffer + 4096) /* model/activation_5/Relu;model/batch_normalization_5/FusedBatchNormV3;model/conv2d_6/BiasAdd/ReadVariableOp/resource;model/conv2d_6/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_6/Conv2D1 */, 8 /* dim_im_out_x */, 8 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 4096) /* model/activation_5/Relu;model/batch_normalization_5/FusedBatchNormV3;model/conv2d_6/BiasAdd/ReadVariableOp/resource;model/conv2d_6/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_6/Conv2D1 */, 1 * 64 * 8 * 8 /* size */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 4096) /* Relu__21:0 */, 8 /* dim_im_in_x */, 8 /* dim_im_in_y */, 64 /* ch_im_in */, const_fold_opt__123 /* const_fold_opt__123 */, 64 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 1 /* padding_x */, 1 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, model_batch_normalization_6_fusedbatchnormv3_model_conv2d_7_biasadd_readvariableop_resource_model_conv2d_7_biasadd /* model/batch_normalization_6/FusedBatchNormV3;model/conv2d_7/BiasAdd/ReadVariableOp/resource;model/conv2d_7/BiasAdd */, 6 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 8192) /* model/batch_normalization_6/FusedBatchNormV3;model/conv2d_7/BiasAdd/ReadVariableOp/resource;model/conv2d_7/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_7/Conv2D */, 8 /* dim_im_out_x */, 8 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_convolve_HWC_q7_fast_nonsquare((scratch_buffer + 19456) /* Relu__19:0 */, 16 /* dim_im_in_x */, 16 /* dim_im_in_y */, 32 /* ch_im_in */, const_fold_opt__127 /* const_fold_opt__127 */, 64 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 2 /* stride_x */, 2 /* stride_y */, model_conv2d_8_biasadd_model_conv2d_8_conv2d_model_conv2d_8_biasadd_readvariableop_resource /* model/conv2d_8/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_8/BiasAdd/ReadVariableOp/resource */, 4 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 0) /* model/conv2d_8/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_8/BiasAdd/ReadVariableOp/resource1 */, 8 /* dim_im_out_x */, 8 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_elementwise_add_s8((scratch_buffer + 0) /* model/conv2d_8/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_8/BiasAdd/ReadVariableOp/resource1 */, (scratch_buffer + 8192) /* model/batch_normalization_6/FusedBatchNormV3;model/conv2d_7/BiasAdd/ReadVariableOp/resource;model/conv2d_7/BiasAdd;model/conv2d_8/Conv2D;model/conv2d_7/Conv2D */, 0 /* input_1_offset */, 2147483647 /* input_1_mult */, -2 /* input_1_shift */, 0 /* input_2_offset */, 1077748608 /* input_2_mult */, 0 /* input_2_shift */, 2 /* left_shift */, (scratch_buffer + 0) /* model/activation_6/Relu;model/add_2/add */, 0 /* out_offset */, 1542219264 /* out_mult */, 0 /* out_shift */, -128 /* out_activation_min */, 127 /* out_activation_max */, (1 * 64 * 8 * 8) /* block_size */);
arm_relu_q7((scratch_buffer + 0) /* model/activation_6/Relu;model/add_2/add */, 1 * 64 * 8 * 8 /* size */);
arm_avepool_q7_HWC((scratch_buffer + 0) /* Relu__26:0 */, 8 /* dim_im_in */, 64 /* ch_im_in */, 8 /* dim_kernel */, 0 /* padding */, 8 /* stride */, 1 /* dim_im_out */, reinterpret_cast<q7_t*>(col_buffer) /* bufferA */, (scratch_buffer + 4096) /* model/average_pooling2d/AvgPool */);

arm_fully_connected_q7((scratch_buffer + 4096) /* model/flatten/Reshape */, const_fold_opt__120 /* const_fold_opt__120 */, 64 /* dim_vec */, 10 /* num_of_rows */, 0 /* bias_shift */, 6 /* out_shift */, model_dense_biasadd_readvariableop_resource /* model/dense/BiasAdd/ReadVariableOp/resource */, (scratch_buffer + 0) /* Add__29:0 */, col_buffer /* vec_buffer */);
arm_softmax_q7((scratch_buffer + 0) /* Add__29:0 */, 1 * 10 /* dim_vec */, (scratch_buffer + 10) /* Identity */);

memcpy(output_data, (scratch_buffer + 10) /* Identity */, (1 * 10));

const struct onnc_output_tensor_t output_tensor = { .data = output_data, .size = (1 * 10) };
return output_tensor;
}
#if defined(__cplusplus)
}
#endif /* defined(__cplusplus) */
