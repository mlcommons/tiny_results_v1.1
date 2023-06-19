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

static const q7_t test57[32] = TEST57;

static const q7_t test56[32] = TEST56;

static const q7_t test55[64] = TEST55;

static const q7_t test54[8] = TEST54;

static const q7_t test53[64 * 1 * 3 * 3] = TEST53;

static const q7_t test52[64 * 32 * 1 * 1] = TEST52;

static const q7_t test51[32 * 1 * 3 * 3] = TEST51;

static const q7_t test50[16 * 8 * 1 * 1] = TEST50;

static const q7_t test19[19 * 1 * 3 * 3] = TEST19;

static const q7_t test18[19] = TEST18;

static const q7_t test17[19 * 19 * 1 * 1] = TEST17;

static const q7_t test16[19] = TEST16;

static const q7_t test15[19 * 1 * 3 * 3] = TEST15;

static const q7_t test14[19] = TEST14;

static const q7_t test13[27 * 19 * 1 * 1] = TEST13;

static const q7_t test12[27] = TEST12;

static const q7_t test11[27 * 1 * 3 * 3] = TEST11;

static const q7_t test10[27] = TEST10;

static const q7_t test49[16] = TEST49;

static const q7_t test48[16] = TEST48;

static const q7_t test47[32] = TEST47;

static const q7_t test46[8 * 1 * 3 * 3] = TEST46;

static const q7_t test45[16 * 1 * 3 * 3] = TEST45;

static const q7_t test44[32 * 32 * 1 * 1] = TEST44;

static const q7_t test43[8 * 3 * 3 * 3] = TEST43;

static const q7_t test42[32 * 16 * 1 * 1] = TEST42;

static const q7_t test41[32] = TEST41;

static const q7_t test40[32 * 1 * 3 * 3] = TEST40;

static const q7_t test39[8] = TEST39;

static const q7_t test38[64] = TEST38;

static const q7_t test37[60 * 64 * 1 * 1] = TEST37;

static const q7_t test36[60] = TEST36;

static const q7_t test35[60 * 1 * 3 * 3] = TEST35;

static const q7_t test34[60] = TEST34;

static const q7_t test33[107 * 60 * 1 * 1] = TEST33;

static const q7_t test32[107] = TEST32;

static const q7_t test31[107 * 1 * 3 * 3] = TEST31;

static const q7_t test30[107] = TEST30;

static const q7_t test9[32 * 27 * 1 * 1] = TEST9;

static const q7_t test8[32] = TEST8;

static const q7_t test7[32 * 1 * 3 * 3] = TEST7;

static const q7_t test6[32] = TEST6;

static const q7_t test5[22 * 32 * 1 * 1] = TEST5;

static const q7_t test4[22] = TEST4;

static const q7_t test2[22 * 2] = TEST2;

static const q7_t test1[2] = TEST1;

static const q7_t test29[58 * 107 * 1 * 1] = TEST29;

static const q7_t test28[58] = TEST28;

static const q7_t test27[58 * 1 * 3 * 3] = TEST27;

static const q7_t test26[58] = TEST26;

static const q7_t test25[30 * 58 * 1 * 1] = TEST25;

static const q7_t test24[30] = TEST24;

static const q7_t test23[30 * 1 * 3 * 3] = TEST23;

static const q7_t test22[30] = TEST22;

static const q7_t test21[19 * 30 * 1 * 1] = TEST21;

static const q7_t test20[19] = TEST20;

static const int mean_data[3] = MEAN_DATA;

static const unsigned scale_data[3] = SCALE_DATA;

static q15_t col_buffer[1926];
static q7_t scratch_buffer[55296];
static q7_t output_data[1 * 2];

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct onnc_input_tensor_t onnc_get_input_tensor() {
const struct onnc_input_tensor_t input_tensor = { .data = (uint8_t*)((scratch_buffer + 0)), .size = (1 * 3 * 96 * 96) };
return input_tensor;
}

struct onnc_output_tensor_t onnc_main() {
#if defined(RTE_Compiler_EventRecorder)
EventRecorderInitialize(EventRecordAll, 1);
#endif /* defined(RTE_Compiler_EventRecorder) */

{
const struct onnc_input_tensor_t input_buffer = onnc_get_input_tensor();
for (size_t pixel = 0; pixel < 1 * 3 * 96 * 96; pixel += 3) {
*((scratch_buffer + 0) + pixel + 0) = (q7_t)(__SSAT(((int)(input_buffer.data[pixel + 0]) - mean_data[0]) >> (scale_data[0]), 8));
*((scratch_buffer + 0) + pixel + 1) = (q7_t)(__SSAT(((int)(input_buffer.data[pixel + 1]) - mean_data[1]) >> (scale_data[1]), 8));
*((scratch_buffer + 0) + pixel + 2) = (q7_t)(__SSAT(((int)(input_buffer.data[pixel + 2]) - mean_data[2]) >> (scale_data[2]), 8));
}
}

arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 0) /* test0 */, 96 /* dim_im_in_x */, 96 /* dim_im_in_y */, 3 /* ch_im_in */, test43 /* test43 */, 8 /* ch_im_out */, 3 /* dim_kernel_x */, 3 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 2 /* stride_x */, 2 /* stride_y */, test39 /* test39 */, 7 /* bias_shift */, 9 /* out_shift */, (scratch_buffer + 27648) /* test58 */, 48 /* dim_im_out_x */, 48 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 27648) /* test58 */, 1 * 8 * 48 * 48 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 27648) /* test59 */, 48 /* dim_im_in */, 8 /* ch_im_in */, test46 /* test46 */, 8 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test54 /* test54 */, 3 /* bias_shift */, 4 /* out_shift */, (scratch_buffer + 0) /* test60 */, 48 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test60 */, 1 * 8 * 48 * 48 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 0) /* test61 */, 48 /* dim_im_in_x */, 48 /* dim_im_in_y */, 8 /* ch_im_in */, test50 /* test50 */, 16 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test49 /* test49 */, 5 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 18432) /* test62 */, 48 /* dim_im_out_x */, 48 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 18432) /* test62 */, 1 * 16 * 48 * 48 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 18432) /* test63 */, 48 /* dim_im_in */, 16 /* ch_im_in */, test45 /* test45 */, 16 /* ch_im_out */, 3 /* dim_kernel */, 0 /* padding */, 2 /* stride */, test48 /* test48 */, 5 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 0) /* test64 */, 24 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test64 */, 1 * 16 * 24 * 24 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 0) /* test65 */, 24 /* dim_im_in_x */, 24 /* dim_im_in_y */, 16 /* ch_im_in */, test42 /* test42 */, 32 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test57 /* test57 */, 5 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 9216) /* test66 */, 24 /* dim_im_out_x */, 24 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 9216) /* test66 */, 1 * 32 * 24 * 24 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 9216) /* test67 */, 24 /* dim_im_in */, 32 /* ch_im_in */, test40 /* test40 */, 32 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test41 /* test41 */, 4 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 27648) /* test68 */, 24 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 27648) /* test68 */, 1 * 32 * 24 * 24 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 27648) /* test69 */, 24 /* dim_im_in_x */, 24 /* dim_im_in_y */, 32 /* ch_im_in */, test44 /* test44 */, 32 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test47 /* test47 */, 5 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test70 */, 24 /* dim_im_out_x */, 24 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test70 */, 1 * 32 * 24 * 24 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test71 */, 24 /* dim_im_in */, 32 /* ch_im_in */, test51 /* test51 */, 32 /* ch_im_out */, 3 /* dim_kernel */, 0 /* padding */, 2 /* stride */, test56 /* test56 */, 5 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 18432) /* test72 */, 12 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 18432) /* test72 */, 1 * 32 * 12 * 12 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 18432) /* test73 */, 12 /* dim_im_in_x */, 12 /* dim_im_in_y */, 32 /* ch_im_in */, test52 /* test52 */, 64 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test55 /* test55 */, 5 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test74 */, 12 /* dim_im_out_x */, 12 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test74 */, 1 * 64 * 12 * 12 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test75 */, 12 /* dim_im_in */, 64 /* ch_im_in */, test53 /* test53 */, 64 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test38 /* test38 */, 3 /* bias_shift */, 5 /* out_shift */, (scratch_buffer + 9216) /* test76 */, 12 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 9216) /* test76 */, 1 * 64 * 12 * 12 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 9216) /* test77 */, 12 /* dim_im_in_x */, 12 /* dim_im_in_y */, 64 /* ch_im_in */, test37 /* test37 */, 60 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test36 /* test36 */, 6 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test78 */, 12 /* dim_im_out_x */, 12 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test78 */, 1 * 60 * 12 * 12 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test79 */, 12 /* dim_im_in */, 60 /* ch_im_in */, test35 /* test35 */, 60 /* ch_im_out */, 3 /* dim_kernel */, 0 /* padding */, 2 /* stride */, test34 /* test34 */, 4 /* bias_shift */, 5 /* out_shift */, (scratch_buffer + 8640) /* test80 */, 6 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 8640) /* test80 */, 1 * 60 * 6 * 6 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 8640) /* test81 */, 6 /* dim_im_in_x */, 6 /* dim_im_in_y */, 60 /* ch_im_in */, test33 /* test33 */, 107 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test32 /* test32 */, 7 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test82 */, 6 /* dim_im_out_x */, 6 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test82 */, 1 * 107 * 6 * 6 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test83 */, 6 /* dim_im_in */, 107 /* ch_im_in */, test31 /* test31 */, 107 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test30 /* test30 */, 5 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 3852) /* test84 */, 6 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 3852) /* test84 */, 1 * 107 * 6 * 6 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 3852) /* test85 */, 6 /* dim_im_in_x */, 6 /* dim_im_in_y */, 107 /* ch_im_in */, test29 /* test29 */, 58 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test28 /* test28 */, 7 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test86 */, 6 /* dim_im_out_x */, 6 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test86 */, 1 * 58 * 6 * 6 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test87 */, 6 /* dim_im_in */, 58 /* ch_im_in */, test27 /* test27 */, 58 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test26 /* test26 */, 3 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 2088) /* test88 */, 6 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 2088) /* test88 */, 1 * 58 * 6 * 6 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 2088) /* test89 */, 6 /* dim_im_in_x */, 6 /* dim_im_in_y */, 58 /* ch_im_in */, test25 /* test25 */, 30 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test24 /* test24 */, 6 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test90 */, 6 /* dim_im_out_x */, 6 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test90 */, 1 * 30 * 6 * 6 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test91 */, 6 /* dim_im_in */, 30 /* ch_im_in */, test23 /* test23 */, 30 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test22 /* test22 */, 4 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 1080) /* test92 */, 6 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 1080) /* test92 */, 1 * 30 * 6 * 6 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 1080) /* test93 */, 6 /* dim_im_in_x */, 6 /* dim_im_in_y */, 30 /* ch_im_in */, test21 /* test21 */, 19 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test20 /* test20 */, 5 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test94 */, 6 /* dim_im_out_x */, 6 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test94 */, 1 * 19 * 6 * 6 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test95 */, 6 /* dim_im_in */, 19 /* ch_im_in */, test19 /* test19 */, 19 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test18 /* test18 */, 4 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 684) /* test96 */, 6 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 684) /* test96 */, 1 * 19 * 6 * 6 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 684) /* test97 */, 6 /* dim_im_in_x */, 6 /* dim_im_in_y */, 19 /* ch_im_in */, test17 /* test17 */, 19 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test16 /* test16 */, 5 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test98 */, 6 /* dim_im_out_x */, 6 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test98 */, 1 * 19 * 6 * 6 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 0) /* test99 */, 6 /* dim_im_in */, 19 /* ch_im_in */, test15 /* test15 */, 19 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test14 /* test14 */, 4 /* bias_shift */, 5 /* out_shift */, (scratch_buffer + 684) /* test100 */, 6 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 684) /* test100 */, 1 * 19 * 6 * 6 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 684) /* test101 */, 6 /* dim_im_in_x */, 6 /* dim_im_in_y */, 19 /* ch_im_in */, test13 /* test13 */, 27 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test12 /* test12 */, 7 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 1368) /* test102 */, 6 /* dim_im_out_x */, 6 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 1368) /* test102 */, 1 * 27 * 6 * 6 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 1368) /* test103 */, 6 /* dim_im_in */, 27 /* ch_im_in */, test11 /* test11 */, 27 /* ch_im_out */, 3 /* dim_kernel */, 0 /* padding */, 2 /* stride */, test10 /* test10 */, 5 /* bias_shift */, 7 /* out_shift */, (scratch_buffer + 0) /* test104 */, 3 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test104 */, 1 * 27 * 3 * 3 /* size */);
arm_convolve_HWC_q7_basic_nonsquare((scratch_buffer + 0) /* test105 */, 3 /* dim_im_in_x */, 3 /* dim_im_in_y */, 27 /* ch_im_in */, test9 /* test9 */, 32 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test8 /* test8 */, 8 /* bias_shift */, 8 /* out_shift */, (scratch_buffer + 243) /* test106 */, 3 /* dim_im_out_x */, 3 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 243) /* test106 */, 1 * 32 * 3 * 3 /* size */);
arm_depthwise_separable_conv_HWC_q7((scratch_buffer + 243) /* test107 */, 3 /* dim_im_in */, 32 /* ch_im_in */, test7 /* test7 */, 32 /* ch_im_out */, 3 /* dim_kernel */, 1 /* padding */, 1 /* stride */, test6 /* test6 */, 5 /* bias_shift */, 5 /* out_shift */, (scratch_buffer + 531) /* test108 */, 3 /* dim_im_out */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 531) /* test108 */, 1 * 32 * 3 * 3 /* size */);
arm_convolve_1x1_HWC_q7_fast_nonsquare((scratch_buffer + 531) /* test109 */, 3 /* dim_im_in_x */, 3 /* dim_im_in_y */, 32 /* ch_im_in */, test5 /* test5 */, 22 /* ch_im_out */, 1 /* dim_kernel_x */, 1 /* dim_kernel_y */, 0 /* padding_x */, 0 /* padding_y */, 1 /* stride_x */, 1 /* stride_y */, test4 /* test4 */, 5 /* bias_shift */, 6 /* out_shift */, (scratch_buffer + 0) /* test110 */, 3 /* dim_im_out_x */, 3 /* dim_im_out_y */, col_buffer /* bufferA */, NULL /* bufferB */);
arm_relu_q7((scratch_buffer + 0) /* test110 */, 1 * 22 * 3 * 3 /* size */);
arm_avepool_q7_HWC((scratch_buffer + 0) /* test111 */, 3 /* dim_im_in */, 22 /* ch_im_in */, 3 /* dim_kernel */, 0 /* padding */, 3 /* stride */, 1 /* dim_im_out */, reinterpret_cast<q7_t*>(col_buffer) /* bufferA */, (scratch_buffer + 198) /* test112 */);

arm_fully_connected_q7((scratch_buffer + 198) /* test113 */, test2 /* test2 */, 22 /* dim_vec */, 2 /* num_of_rows */, 4 /* bias_shift */, 6 /* out_shift */, test1 /* test1 */, (scratch_buffer + 0) /* test115 */, col_buffer /* vec_buffer */);
arm_softmax_q7((scratch_buffer + 0) /* test115 */, 1 * 2 /* dim_vec */, (scratch_buffer + 2) /* test116 */);

memcpy(output_data, (scratch_buffer + 2) /* test116 */, (1 * 2));

const struct onnc_output_tensor_t output_tensor = { .data = output_data, .size = (1 * 2) };
return output_tensor;
}
#if defined(__cplusplus)
}
#endif /* defined(__cplusplus) */
