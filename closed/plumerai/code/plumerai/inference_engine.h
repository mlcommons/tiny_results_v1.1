// Copyright (C) 2023, Plumerai Ltd.
// All rights reserved.
#ifndef PLUMERAI_INFERENCE_ENGINE_H
#define PLUMERAI_INFERENCE_ENGINE_H

#include <cstdint>

#include "tensorflow_compatibility.h"

namespace plumerai {

class InferenceEngine {
 public:
  InferenceEngine() : impl_(nullptr){};
  ~InferenceEngine();

  // Initialize the InferenceEngine. The tensor arena has to be provided by the
  // user and should be large enough to hold the model's activation tensors. For
  // best performance the tensor arena is 16-byte aligned. This must be done
  // once at the very start before calling any other function. If `report_mode`
  // is true then the `profiler` argument is ignored and a profiler is allocated
  // by the engine itself in the arena and `print_report` becomes available. In
  // case the library was built to include multiple models, `model_id` can be
  // used to select a model for this instance of the `InferenceEngine` class.
  template <bool report_mode = false>
  TfLiteStatus Initialize(std::uint8_t* tensor_arena_ptr, int tensor_arena_size,
                          int model_id = 0,
                          ::tflite::MicroProfilerInterface* profiler = nullptr);
  template <bool report_mode = false>
  TfLiteStatus Initialize(std::uint8_t** tensor_arena_ptrs,
                          int* tensor_arena_sizes, int num_tensor_arenas,
                          int model_id = 0,
                          ::tflite::MicroProfilerInterface* profiler = nullptr);

  // Initialize the InferenceEngine using the advanced arena setup. Same as
  // above, but now the tensor arena can be split in two parts:
  // The persistent arena stores persistent data such as tensor metadata and
  // stateful LSTM tensors. The non-persistent arena is only used during
  // `Invoke` and can be re-used by the user or by another model after `Invoke`
  // completes. See the documentation for more information.
  template <bool report_mode = false>
  TfLiteStatus Initialize(std::uint8_t* persistent_tensor_arena_ptr,
                          int persistent_tensor_arena_size,
                          std::uint8_t* non_persistent_tensor_arena_ptr,
                          int non_persistent_tensor_arena_size,
                          int model_id = 0,
                          ::tflite::MicroProfilerInterface* profiler = nullptr);
  template <bool report_mode = false>
  TfLiteStatus Initialize(std::uint8_t* persistent_tensor_arena_ptr,
                          int persistent_tensor_arena_size,
                          std::uint8_t** non_persistent_tensor_arena_ptrs,
                          int* non_persistent_tensor_arena_sizes,
                          int num_non_persistent_tensor_arenas,
                          int model_id = 0,
                          ::tflite::MicroProfilerInterface* profiler = nullptr);

  // Allocates input, output and intermediate tensors in the tensor arena. This
  // needs to be called before running inference with `Invoke`.
  TfLiteStatus AllocateTensors();

  // Run inference assuming input data is already set using `input` below.
  TfLiteStatus Invoke();

  // Access the input and output tensors.
  TfLiteTensor* input(int input_id);
  TfLiteTensor* output(int output_id);

  // Query the number of input and output tensors.
  size_t inputs_size() const;
  size_t outputs_size() const;

  // When `report_mode` is set to `true` in `Initialize`, this method can print
  // the report. It needs to be called after `Invoke` is called.
  void print_report() const;

  // Reset the state to be what you would expect when the interpreter is first
  // created after `AllocateTensors` is called. This is useful for recurrent
  // neural networks (e.g. LSTMs) which can preserve internal state between
  // `Invoke` calls.
  TfLiteStatus Reset();

  // `AddCustomOp` is only needed if there are custom ops not supported by
  // the inference engine. It has to be called before `AllocateTensors`.
  TfLiteStatus AddCustomOp(const char* name, TfLiteRegistration* registration);

  // This method gives the optimal arena size, i.e. the size that was actually
  // needed. It is only available after `AllocateTensors` has been called.
  size_t arena_used_bytes() const;

  class impl;

 private:
  impl* impl_;
};

}  // namespace plumerai

#endif  // PLUMERAI_INFERENCE_ENGINE_H
