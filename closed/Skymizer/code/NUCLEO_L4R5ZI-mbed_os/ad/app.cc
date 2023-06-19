//===----------------------------------------------------------------------===//
//
// Copyright(c) 2021, Skymizer Taiwan Inc.
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "app.h"
#include "onnc_main.h"
#include "mbed.h"
#include "micro_model_setting.h"
#include "api/internally_implemented.h"
#include <stdint.h>
#include <math.h>

#define note(...) printf(__VA_ARGS__)
#define note_failed() printf("\u001b[31mFailed\u001b[0m\r\n")
#define note_ok() printf("\u001b[32mOK\u001b[0m\r\n")

static int8_t g_result[kOutputSize];
static int8_t input_data[kInputSize];
static float raw_data[kInputSize];
InOutDescriptor in_out_desc;

void final_initialize() {
  // Setup the descriptor for the testbench.
  in_out_desc.in_buf = input_data;
  in_out_desc.in_bytes = kInputSize * sizeof(int8_t);
  in_out_desc.out_buf = g_result;
  in_out_desc.out_bytes = sizeof(g_result);
}

void infer() {
  onnc_input_tensor_t in_buf = onnc_get_input_tensor();
  memcpy(in_buf.data, input_data, in_buf.size);
  onnc_output_tensor_t result = onnc_main();
  memcpy(g_result, result.data, result.size);
}

void load_tensor() {
  float in_sf, out_sf;
  onnc_get_io_scaling_factors(&in_sf, &out_sf);
  ee_get_buffer(reinterpret_cast<uint8_t *>(raw_data),
                kInputSize * sizeof(float));
  for (size_t i = 0; i < kFeatureSize; i++) {
    reinterpret_cast<int8_t *>(in_out_desc.in_buf)[i] = (int8_t)(roundf(raw_data[i] * in_sf));
  }
}

static float calculate_result() {
  size_t feature_size = kFeatureSize;
  float diffsum = 0;
  float in_sf, out_sf;
  onnc_get_io_scaling_factors(&in_sf, &out_sf);

  for (size_t i = 0; i < feature_size; i++) {
    float diff = ((float)g_result[i] * out_sf) - raw_data[i];
    diffsum += diff * diff;
  }
  diffsum /= feature_size;

  return diffsum;
}

void results() {
  note("Done inference ...");
  note_ok();
  float result = calculate_result();
  int tmp0 = result;
  int tmp1 = (int)(result * 10)%10;
  int tmp2 = (int)(result * 100)%10;
  int tmp3 = (int)(result * 1000)%10;
  note("m-results-[%d.%d%d%d]\r\n", tmp0, tmp1, tmp2, tmp3);
}
