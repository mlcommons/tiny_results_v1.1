#ifndef DMA_UTIL_H_
#define DMA_UTIL_H_

#include <stdio.h>
#include "xaxidma.h"

int init_dma(XAxiDma* dma, int DMA_ID) ;
int transfer_in(XAxiDma* dma_in, UINTPTR* in, u32 size);
int transfer_both(XAxiDma* dma_in, XAxiDma* dma_out, UINTPTR* in, UINTPTR* out, u32 size_in, u32 size_out);

#endif
