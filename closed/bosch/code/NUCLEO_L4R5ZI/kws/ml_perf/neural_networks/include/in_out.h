/**
This file was generated by
   _   _      _      _       _____
  | | | |    / \    | |     | ____|
  | |_| |   / _ \   | |     |  _|
  |  _  |  / ___ \  | |___  | |___
  |_| |_| /_/   \_\ |_____| |_____|
version: 1.0.8
Model: kws_ref_model.tflite
Target: CortexM4
Timestamp: 2023_05_16_11.18.08
Copyright © 2023 Robert Bosch GmbH
**/
#ifndef IN_OUT_H
#define IN_OUT_H

#include <stdint.h>
#include <math.h>

#define BUFFER_SIZE 16576

// Input sizes
#define KWS_REF_MODEL_INPUT_1_SIZE 490

// Output sizes
#define KWS_REF_MODEL_IDENTITY_SIZE 12

// Type definitions
typedef int8_t kws_ref_model_input_1_type[1][49][10][1];
typedef int8_t kws_ref_model_Identity_type[1][12];

typedef struct {
	int8_t* input_1;
	int8_t* Identity;
} kws_ref_modelInOut;

/// Get the input and expected output for the neural network
///
/// This sets the pointers in the struct defined above to the correct places in the buffer
/// \param buffer Buffer that contains the memory the neural network operates on
/// return Struct containing pointers to the network input and output
kws_ref_modelInOut get_kws_ref_modelInOut_for(char* buffer);

/// Quantizes a single value from input input_1 of network kws_ref_model
/// /param value Value to quantize
/// return Quantized value
int8_t kws_ref_model_input_1_quantize_float_to_int8(float value);

/// Dequantizes a single value from output Identity of network kws_ref_model
/// /param value Value to dequantize
/// return Dequantized value
float kws_ref_model_Identity_dequantize_int8_to_float(int8_t value);

#endif // IN_OUT_H