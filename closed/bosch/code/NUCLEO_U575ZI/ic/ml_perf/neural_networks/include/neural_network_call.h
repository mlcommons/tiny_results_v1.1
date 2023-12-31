/**
This file was generated by
   _   _      _      _       _____
  | | | |    / \    | |     | ____|
  | |_| |   / _ \   | |     |  _|
  |  _  |  / ___ \  | |___  | |___
  |_| |_| /_/   \_\ |_____| |_____|
version: 1.0.8
Model: pretrainedResnet_quant.tflite
Target: CortexM33DSP
Timestamp: 2023_05_15_11.26.00
Copyright © 2023 Robert Bosch GmbH
**/
#ifndef NEURAL_NETWORK_CALL_H
#define NEURAL_NETWORK_CALL_H
#ifdef __cplusplus
extern "C"{
#endif
#include <stdint.h>

#include "in_out.h"

/// Call the neural network 
/// \param buffer Memory buffer to use for the neural network call
void pretrainedResnet_quant_call(char buffer[BUFFER_SIZE]);

#ifdef __cplusplus
}
#endif
#endif // NEURAL_NETWORK_CALL_H
