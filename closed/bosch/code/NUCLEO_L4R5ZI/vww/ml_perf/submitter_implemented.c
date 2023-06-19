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

#include "api/submitter_implemented.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/internally_implemented.h"

#include "in_out.h"
#include "neural_network_call.h"
#include "tim.h"

//UnbufferedSerial pc(USBTX, USBRX);


int timestampPin = 0;

char buffer[BUFFER_SIZE];
vww_96_int8InOut vww_96_int8_in_out;


void init_submitter() {
	vww_96_int8_in_out = get_vww_96_int8InOut_for(buffer);
}

void invoke() {
	vww_96_int8_call(buffer);

}

void set_input(const int8_t *custom_input) {
	memcpy(vww_96_int8_in_out.input_1_int8, custom_input, 27648);
}

int8_t *get_output() {
	return vww_96_int8_in_out.Identity_int8;
}

float dequantize(const int8_t value) {
	return vww_96_int8_Identity_int8_dequantize_int8_to_float(value);
}

void th_load_tensor() {
    int8_t input[kInputSize];
    size_t bytes = ee_get_buffer((uint8_t *)(input),
                                 kInputSize * sizeof(int8_t));

    for (size_t i = 0; i < bytes; i++) {
    	input[i] -= 128;
    }

    th_printf("bytes %d\n", bytes);

    set_input(input);

}


// Add to this method to return real inference results.
void th_results() {
    uint8_t* output = get_output();

    th_printf("m-results-[");

    for (int i = 0; i < kCategoryCount; i++) {
        float converted = dequantize(output[i]);
        th_printf("%0.3f", converted);
        if (i < kCategoryCount - 1) {
            th_printf(",");
        }
    }

    th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() { invoke(); }

/// \brief optional API.
void th_final_initialize(void) {

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
    //pc.baud(115200);
#endif
}

void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE == 1
    timestampPin = 0;
    for (int i = 0; i < 100000; ++i) {
      asm("nop");
    }
    timestampPin = 1;
#else
    unsigned long microSeconds = 0ul;
    /* USER CODE 2 BEGIN */
    //microSeconds = us_ticker_read();
    microSeconds = __HAL_TIM_GET_COUNTER(&htim5);
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
