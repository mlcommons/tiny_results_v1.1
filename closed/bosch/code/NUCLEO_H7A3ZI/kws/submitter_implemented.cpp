/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
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

#include "api/submitter_implemented.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "api/internally_implemented.h"
#include "mbed.h"
#include "hale_runner.h"

UnbufferedSerial pc(USBTX, USBRX);

HaleRunner *runner;

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {

    int8_t input[kInputSize];
    size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input),
                                 kInputSize * sizeof(int8_t));
    runner->SetInput(input);
}

// Add to this method to return real inference results.
void th_results() {
    auto output = runner->GetOutput();

    th_printf("m-results-[");

    for (size_t i = 0; i < kCategoryCount; i++) {
        float converted = runner->dequantize(output[i]);
        th_printf("0.%d", static_cast<int>(converted * 10));
        th_printf("%d", static_cast<int>(converted * 100) % 10);
        th_printf("%d", static_cast<int>(converted * 1000) % 10);
        if (i < kCategoryCount - 1) {
            th_printf(",");
        }
    }

    th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() { runner->Invoke(); }

/// \brief optional API.
void th_final_initialize(void) {
    static HaleRunner model_runner;
    runner = &model_runner;
}

void th_pre() {}

void th_post() {}

void th_command_ready(char volatile *p_command) {
    p_command = p_command;
    ee_serial_command_parser_callback((char *) p_command);
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
    (void) th_vprintf(p_fmt, args); /* ignore return */
    va_end(args);
}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE == 1
    pc.baud(9600);
#else
    pc.baud(115200);
#endif
}

void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE == 1
    timestampPin = 0;
    for (int i = 0; i < 100'000; ++i) {
      asm("nop");
    }
    timestampPin = 1;
#else
    unsigned long microSeconds = 0ul;
    /* USER CODE 2 BEGIN */
    microSeconds = us_ticker_read();
    /* USER CODE 2 END */
    /* This message must NOT be changed. */
    th_printf(EE_MSG_TIMESTAMP, microSeconds);
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
    th_timestamp();
}
