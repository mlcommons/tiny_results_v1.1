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
/// \brief C++ implementations of submitter_implemented.h
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

#include "api/submitter_implemented.h"
#include "api/internally_implemented.h"
#include "mbed.h"
#include "app.h"
Timer timer;

Serial pc(USBTX, USBRX);

DigitalOut timestampPin(D7);
unsigned char cmd_buf[32+640];

#define fatal(...)                                                             \
  printf("Fatal: " __VA_ARGS__);                                               \
  return 0
#define verbose(...)                                                           \
  if (MBED_CONF_APP_VERBOSE)                                                   \
  printf(__VA_ARGS__)

void th_load_tensor(){load_tensor();}

void th_results() {
  // step 3: get logits from result buffer
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

void th_command_ready(char volatile *p_command) {
  fflush(stdout);
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() {
  return pc.getc();
}

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE==1
  pc.baud(9600);
  printf("baud :9600\n");
#else
  pc.baud(115200);
  printf("baud :115200\n");
#endif

}

void th_timestamp(void) {
 # if EE_CFG_ENERGY_MODE==1
  timestampPin = 0;
  for (int i=0; i<100'000; ++i) {
    asm("nop");
  }
  timestampPin = 1;
 #else
  /* USER CODE 2 BEGIN */
  unsigned long ms_cum = timer.read_us();
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, ms_cum);
  #endif
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  timer.start();
  th_timestamp();
}