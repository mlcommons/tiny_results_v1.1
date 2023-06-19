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
        ad01_int8_in_out = get_ad01_int8InOut_for(buffer);
    }

    void Invoke() {
        ad01_int8_call(buffer);

    }

    void SetInput(const int8_t *custom_input) {
        memcpy(ad01_int8_in_out.input_1, custom_input, 640);
    }

    int8_t *GetOutput() {
        return ad01_int8_in_out.Identity;
    }

    float dequantize(const int8_t value) {
        return ad01_int8_Identity_dequantize_int8_to_float(value);
    }

    int8_t quantize(const float value) {
        return ad01_int8_input_1_quantize_float_to_int8(value);
    }

private:
    char buffer[BUFFER_SIZE];
    ad01_int8InOut ad01_int8_in_out;

};
