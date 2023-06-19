/*
Copyright (C) EEMBC(R). All Rights Reserved
All EEMBC Benchmark Software are products of EEMBC and are provided under the
terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
are proprietary intellectual properties of EEMBC and its Members and is
protected under all applicable laws, including all applicable copyright laws.
If you received this EEMBC Benchmark Software without having a currently
effective EEMBC Benchmark License Agreement, you must discontinue use.
Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C implementations of submitter_implemented.h

#include "api/submitter_implemented.h"
#include "api/internally_implemented.h"
#include "onnc_main.h"
#include "micro_model_setting.h"

#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <unistd.h>

#include <stdint.h>
#include <stddef.h>
#include <math.h>

typedef struct {
  void *in_buf;
  size_t in_bytes;
  void *out_buf;
  size_t out_bytes;
} InOutDescriptor;


static int8_t g_result[kOutputSize];
static int8_t input_data[kInputSize];
static float raw_data[kInputSize];
InOutDescriptor in_out_desc;

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
  float result = calculate_result();
  int tmp0 = result;
  int tmp1 = (int)(result * 10)%10;
  int tmp2 = (int)(result * 100)%10;
  int tmp3 = (int)(result * 1000)%10;
  th_printf("m-results-[%d.%d%d%d]\r\n", tmp0, tmp1, tmp2, tmp3);
}

void infer() {
  onnc_input_tensor_t in_buf = onnc_get_input_tensor();
  memcpy(in_buf.data, input_data, in_buf.size);
  onnc_output_tensor_t result = onnc_main();
  memcpy(g_result, result.data, result.size);
}

void final_initialize() {
  onnc_input_tensor_t in_buf = onnc_get_input_tensor();
  onnc_output_tensor_t out_buf = onnc_main();
  // Setup the descriptor for the testbench.
  in_out_desc.in_buf = input_data;
  in_out_desc.in_bytes = sizeof(input_data);
  in_out_desc.out_buf = g_result;
  in_out_desc.out_bytes = sizeof(g_result);

}

#define UART_DEFAULT_BAUDRATE  115200
static const struct device* g_uart;

size_t formatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

uint32_t writeSerial(const char* data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    uart_poll_out(g_uart, data[i]);
  }
  return size;
}

char uartRxRead() {
  unsigned char c;
  int ret = -1;
  while(ret != 0) {
    ret = uart_poll_in(g_uart, &c);
  }
  return (char)c;
}

void uartInit(uint32_t baudrate = UART_DEFAULT_BAUDRATE) {
  g_uart = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
  if (!device_is_ready(g_uart)) {
    printk("uart devices not ready\n");
    return;
  }
  const struct uart_config config = {.baudrate = baudrate,
                                     .parity = UART_CFG_PARITY_NONE,
                                     .stop_bits = UART_CFG_STOP_BITS_1,
                                     .data_bits = UART_CFG_DATA_BITS_8,
                                     .flow_ctrl = UART_CFG_FLOW_CTRL_NONE};
  uart_configure(g_uart, &config);
}


#if EE_CFG_ENERGY_MODE == 1
// use GPIO PC6 which is on connector CN7 pin 1 on the nucleo_l4r5zi
static const char* g_gpio_device_name = "GPIOC";
static const struct device *g_gpio_dev;
static const gpio_pin_t g_gpio_pin = 6;
#endif

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  load_tensor();
}


// Add to this method to return real inference results.
void th_results() {
  results();
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() {
  infer();
}

/// \brief optional API.
void th_final_initialize(void) {
  final_initialize();
}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile* p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char*)p_command);
}

// th_libc implementations.
int th_strncmp(const char* str1, const char* str2, size_t n) { return strncmp(str1, str2, n); }

char* th_strncpy(char* dest, const char* src, size_t n) { return strncpy(dest, src, n); }

size_t th_strnlen(const char* str, size_t maxlen) { return strlen(str); }

char* th_strcat(char* dest, const char* src) { return strcat(dest, src); }

char* th_strtok(char* str1, const char* sep) { return strtok(str1, sep); }

int th_atoi(const char* str) { return atoi(str); }

void* th_memset(void* b, int c, size_t len) { return memset(b, c, len); }

void* th_memcpy(void* dst, const void* src, size_t n) { return memcpy(dst, src, n); }

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char* format, va_list ap) { return vprintf(format, ap); }


void th_printf(const char* p_fmt, ...) {
  char buffer[128];
  int size;
  va_list args;
  va_start(args, p_fmt);
  size = formatMessage(buffer, 128, p_fmt, args);
  va_end(args);
  writeSerial(buffer, (size_t)size);
}

char th_getchar() {
  return uartRxRead();
}

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE == 1
  uartInit(9600);
#else
  uartInit();
#endif
}

void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE == 1
  /* USER CODE 1 BEGIN */
  /* Step 1. Pull pin low */
  gpio_pin_set(g_gpio_dev, g_gpio_pin, 0);
  /* Step 2. Hold low for at least 1us */
  k_busy_wait(1);
  /* Step 3. Release driver */
  gpio_pin_set(g_gpio_dev, g_gpio_pin, 1);
  /* USER CODE 1 END */
#else
  /* USER CODE 2 BEGIN */
  unsigned long microSeconds = (unsigned long)(k_uptime_get() * 1000LL);
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
#endif
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
#if EE_CFG_ENERGY_MODE == 1
  g_gpio_dev = device_get_binding("GPIOC");
  if (g_gpio_dev == NULL) {
    th_printf("GPIO device init failed\r\n");
    return;
  }

  int ret = gpio_pin_configure(g_gpio_dev, g_gpio_pin, GPIO_OUTPUT_HIGH);
  if (ret < 0) {
    th_printf("GPIO pin configure failed\r\n");
    return;
  }
#endif

  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
