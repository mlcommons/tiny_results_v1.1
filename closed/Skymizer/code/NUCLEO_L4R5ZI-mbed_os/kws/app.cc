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

#define note(...) printf(__VA_ARGS__)
#define note_failed() printf("\u001b[31mFailed\u001b[0m\r\n")
#define note_ok() printf("\u001b[32mOK\u001b[0m\r\n")

static int8_t g_result[kOutputSize];
static int8_t input_data[kInputSize];
InOutDescriptor in_out_desc;

void final_initialize() {
  // Setup the descriptor for the testbench.
  in_out_desc.in_buf = input_data;
  in_out_desc.in_bytes = kInputSize * sizeof(uint8_t);
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
  // copy data from gp_buff into input_float
  ee_get_buffer(reinterpret_cast<uint8_t *>(in_out_desc.in_buf),
                kInputSize * sizeof(uint8_t));
}

void results() {
  // copy data from gp_buff into input_float
  note("Done inference ...");
  note_ok();
  note("m-results-[");
  size_t tmp = (12 < in_out_desc.out_bytes) ? 12:in_out_desc.out_bytes;
  for (size_t i = 0; i < tmp; ++i) {
    note("%d", static_cast<int8_t *>(in_out_desc.out_buf)[i]);
    if (i != (tmp-1))
      printf(",");
  }
  note("]\r\n");
}
