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

#ifndef PLUMERAI_TENSORFLOW_COMPATIBILITY_H
#define PLUMERAI_TENSORFLOW_COMPATIBILITY_H
// This file contains only the relevant types and functions that we require from
// TF Micro
//
// C types, required to define `PlumeraiInference`
// From `tensorflow/lite/c/c_api_types.h`
//  - TfLiteStatus
//  - TfLiteType
//  - TfLiteQuantizationParams
// From `tensorflow/lite/c/common.h`
//  - TfLiteTensor
//      - TfLiteIntArray
//      - TfLitePtrUnion
//           - TfLiteComplex64
//           - TfLiteComplex128
//           - TfLiteFloat16
//      - TfLiteQuantizationType
//      - TfLiteQuantization
//      - TfLiteAllocationType
//  - TfLiteRegistration
//
// C++ functions, required to access tensor data
// From `tensorflow/lite/micro/micro_log.h`
//  - ::MicroPrintf
// From `tensorflow/lite/kernels/internal/tensor_ctypes.h`
//  - tflite::GetTensorData<T>
//
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
//
// From `tensorflow/lite/c/c_api_types.h`
//
typedef enum TfLiteStatus {
  kTfLiteOk = 0,
  // Generally referring to an error in the runtime (i.e. interpreter)
  kTfLiteError = 1,
  // Generally referring to an error from a TfLiteDelegate itself.
  kTfLiteDelegateError = 2,
  // Generally referring to an error in applying a delegate due to
  // incompatibility between runtime and delegate, e.g., this error is returned
  // when trying to apply a TF Lite delegate onto a model graph that's already
  // immutable.
  kTfLiteApplicationError = 3,
  // Generally referring to serialized delegate data not being found.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataNotFound = 4,
  // Generally referring to data-writing issues in delegate serialization.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataWriteError = 5,
  // Generally referring to data-reading issues in delegate serialization.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataReadError = 6,
  // Generally referring to issues when the TF Lite model has ops that cannot be
  // resolved at runtime. This could happen when the specific op is not
  // registered or built with the TF Lite framework.
  kTfLiteUnresolvedOps = 7,
  // Generally referring to invocation cancelled by the user.
  // See `interpreter::Cancel`.
  // TODO(b/194915839): Implement `interpreter::Cancel`.
  // TODO(b/250636993): Cancellation triggered by `SetCancellationFunction`
  // should also return this status code.
  kTfLiteCancelled = 8,
} TfLiteStatus;
// Types supported by tensor
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
} TfLiteType;
typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;
//
// From `tensorflow/lite/c/common.h`
//
// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct TfLiteIntArray {
  int size;

#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  int data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  // gcc 6.1+ have a bug where flexible members aren't properly handled
  // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;
// Single-precision complex data type compatible with the C99 definition.
typedef struct TfLiteComplex64 {
  float re, im;  // real and imaginary parts, respectively.
} TfLiteComplex64;
// Double-precision complex data type compatible with the C99 definition.
typedef struct TfLiteComplex128 {
  double re, im;  // real and imaginary parts, respectively.
} TfLiteComplex128;
// Half precision data type compatible with the C99 definition.
typedef struct TfLiteFloat16 {
  uint16_t data;
} TfLiteFloat16;
// SupportedQuantizationTypes.
typedef enum TfLiteQuantizationType {
  // No quantization.
  kTfLiteNoQuantization = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to TfLiteAffineQuantization.
  kTfLiteAffineQuantization = 1,
} TfLiteQuantizationType;
// Structure specifying the quantization used by the tensor, if-any.
typedef struct TfLiteQuantization {
  // The type of quantization held by params.
  TfLiteQuantizationType type;
  // Holds an optional reference to a quantization param structure. The actual
  // type depends on the value of the `type` field (see the comment there for
  // the values and corresponding types).
  void* params;
} TfLiteQuantization;
/* A union of pointers that points to memory for a given tensor. */
typedef union TfLitePtrUnion {
  /* Do not access these members directly, if possible, use
   * GetTensorData<TYPE>(tensor) instead, otherwise only access .data, as other
   * members are deprecated. */
  int32_t* i32;
  uint32_t* u32;
  int64_t* i64;
  uint64_t* u64;
  float* f;
  TfLiteFloat16* f16;
  double* f64;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  TfLiteComplex64* c64;
  TfLiteComplex128* c128;
  int8_t* int8;
  /* Only use this member. */
  void* data;
} TfLitePtrUnion;
// Memory allocation strategies.
//  * kTfLiteMmapRo: Read-only memory-mapped data, or data externally allocated.
//  * kTfLiteArenaRw: Arena allocated with no guarantees about persistence,
//        and available during eval.
//  * kTfLiteArenaRwPersistent: Arena allocated but persistent across eval, and
//        only available during eval.
//  * kTfLiteDynamic: Allocated during eval, or for string tensors.
//  * kTfLitePersistentRo: Allocated and populated during prepare. This is
//        useful for tensors that can be computed during prepare and treated
//        as constant inputs for downstream ops (also in prepare).
//  * kTfLiteCustom: Custom memory allocation provided by the user. See
//        TfLiteCustomAllocation below.
typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
  kTfLitePersistentRo,
  kTfLiteCustom,
} TfLiteAllocationType;
typedef struct TfLiteTensor {
  // Quantization information. Replaces params field above.
  TfLiteQuantization quantization;
  // Quantization information.
  TfLiteQuantizationParams params;
  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;
  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have. NOTE: the product of elements of `dims`
  // and the element datatype size should be equal to `bytes` below.
  TfLiteIntArray* dims;
  // The number of bytes required to store the data of this Tensor. I.e.
  // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  // type is kTfLiteFloat32 and dims = {3, 2} then
  // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;
  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;
  // How memory is mapped
  //  kTfLiteMmapRo: Memory mapped read only.
  //  i.e. weights
  //  kTfLiteArenaRw: Arena allocated read write memory
  //  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;
  // True if the tensor is a variable.
  bool is_variable;
} TfLiteTensor;

typedef struct TfLiteContext TfLiteContext;
typedef struct TfLiteNode TfLiteNode;
typedef struct TfLiteRegistrationExternal TfLiteRegistrationExternal;

typedef struct TfLiteRegistration {
  // Initializes the op from serialized data.
  // Called only *once* for the lifetime of the op, so any one-time allocations
  // should be made here (unless they depend on tensor sizes).
  //
  // If a built-in op:
  //   `buffer` is the op's params data (TfLiteLSTMParams*).
  //   `length` is zero.
  // If custom op:
  //   `buffer` is the op's `custom_options`.
  //   `length` is the size of the buffer.
  //
  // Returns a type-punned (i.e. void*) opaque data (e.g. a primitive pointer
  // or an instance of a struct).
  //
  // The returned pointer will be stored with the node in the `user_data` field,
  // accessible within prepare and invoke functions below.
  // NOTE: if the data is already in the desired format, simply implement this
  // function to return `nullptr` and implement the free function to be a no-op.
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);

  // The pointer `buffer` is the data previously returned by an init invocation.
  void (*free)(TfLiteContext* context, void* buffer);

  // prepare is called when the inputs this node depends on have been resized.
  // context->ResizeTensor() can be called to request output tensors to be
  // resized.
  // Can be called multiple times for the lifetime of the op.
  //
  // Returns kTfLiteOk on success.
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);

  // Execute the node (should read node->inputs and output to node->outputs).
  // Returns kTfLiteOk on success.
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);

  // profiling_string is called during summarization of profiling information
  // in order to group executions together. Providing a value here will cause a
  // given op to appear multiple times is the profiling report. This is
  // particularly useful for custom ops that can perform significantly
  // different calculations depending on their `user-data`.
  const char* (*profiling_string)(const TfLiteContext* context,
                                  const TfLiteNode* node);

  // Builtin codes. If this kernel refers to a builtin this is the code
  // of the builtin. This is so we can do marshaling to other frameworks like
  // NN API.
  // Note: It is the responsibility of the registration binder to set this
  // properly.
  int32_t builtin_code;

  // Custom op name. If the op is a builtin, this will be null.
  // Note: It is the responsibility of the registration binder to set this
  // properly.
  // WARNING: This is an experimental interface that is subject to change.
  const char* custom_name;

  // The version of the op.
  // Note: It is the responsibility of the registration binder to set this
  // properly.
  int version;

  // The external version of `TfLiteRegistration`. Since we can't use internal
  // types (such as `TfLiteContext`) for C API to maintain ABI stability.
  // C API user will provide `TfLiteRegistrationExternal` to implement custom
  // ops. We keep it inside of `TfLiteRegistration` and use it to route
  // callbacks properly.
  TfLiteRegistrationExternal* registration_external;
} TfLiteRegistration;
#ifdef __cplusplus
}  // extern C
#endif

#ifdef __cplusplus
//
// From `tensorflow/lite/micro/micro_log.h`
//
void MicroPrintf(const char* format, ...);
//
// From `tensorflow/lite/kernels/internal/tensor_ctypes.h`
//

namespace tflite {
template <typename T>
inline T* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<T*>(tensor->data.raw) : nullptr;
}
template <typename T>
inline const T* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<const T*>(tensor->data.raw)
                           : nullptr;
}

//
// From `tensorflow/lite/micro/micro_profiler_interface.h`
//
class MicroProfilerInterface;
}  // namespace tflite

#endif  // __cplusplus
#endif  // PLUMERAI_TENSORFLOW_COMPATIBILITY_H
