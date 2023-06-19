//===----------------------------------------------------------------------===//
//
// Copyright(c) 2021, Skymizer Taiwan Inc.
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <stddef.h>

typedef struct {
  void *in_buf;
  size_t in_bytes;
  void *out_buf;
  size_t out_bytes;
} InOutDescriptor;

void final_initialize();
void infer();
void load_tensor();
void results();
