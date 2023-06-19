/**
This file was generated by
   _   _      _      _       _____
  | | | |    / \    | |     | ____|
  | |_| |   / _ \   | |     |  _|
  |  _  |  / ___ \  | |___  | |___
  |_| |_| /_/   \_\ |_____| |_____|
version: 1.0.6
Model: vww_96_int8.tflite
Target: CortexM7
Timestamp: 2023_05_09_11.01.10
Copyright © 2023 Robert Bosch GmbH
**/
#ifndef IN_OUT_H
#define IN_OUT_H

#include <stdint.h>
#include <math.h>

#define BUFFER_SIZE 55296

// Input sizes
#define VWW_96_INT8_INPUT_1_INT8_SIZE 27648

// Output sizes
#define VWW_96_INT8_IDENTITY_INT8_SIZE 2

// Type definitions
typedef int8_t vww_96_int8_input_1_int8_type[1][96][96][3];
typedef int8_t vww_96_int8_Identity_int8_type[1][2];

typedef struct {
	int8_t* input_1_int8;
	int8_t* Identity_int8;
} vww_96_int8InOut;

/// Get the input and expected output for the neural network
///
/// This sets the pointers in the struct defined above to the correct places in the buffer
/// \param buffer Buffer that contains the memory the neural network operates on
/// return Struct containing pointers to the network input and output
vww_96_int8InOut get_vww_96_int8InOut_for(char* buffer);

/// Quantizes a single value from input input_1_int8 of network vww_96_int8
/// /param value Value to quantize
/// return Quantized value
int8_t vww_96_int8_input_1_int8_quantize_float_to_int8(float value);

/// Dequantizes a single value from output Identity_int8 of network vww_96_int8
/// /param value Value to dequantize
/// return Dequantized value
float vww_96_int8_Identity_int8_dequantize_int8_to_float(int8_t value);

#endif // IN_OUT_H
