/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MICRO_MODEL_SETTINGS_H
#define MICRO_MODEL_SETTINGS_H


constexpr int kSlices = 5;
constexpr int kFrequency = 128;
constexpr int kFeatureSize = kSlices * kFrequency;

constexpr int kInputSize = kFeatureSize;
constexpr int kOutputSize = kInputSize;

#endif
