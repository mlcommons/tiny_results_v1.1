/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

extern "C" {
#include "in_out.h"
#include "neural_network_call.h"
}

class HaleRunner {
public:
    explicit HaleRunner() {
        pretrainedResnet_quant_in_out = get_pretrainedResnet_quantInOut_for(buffer);
    }

    void Invoke() {
        pretrainedResnet_quant_call(buffer);

    }

    void SetInput(const int8_t *custom_input) {
        memcpy(pretrainedResnet_quant_in_out.input_1_int8, custom_input, 3072);
    }

    int8_t *GetOutput() {
        return pretrainedResnet_quant_in_out.Identity_int8;
    }

    float dequantize(const int8_t value) {
        return pretrainedResnet_quant_Identity_int8_dequantize_int8_to_float(value);
    }

private:
    char buffer[BUFFER_SIZE];
    pretrainedResnet_quantInOut pretrainedResnet_quant_in_out;

};
