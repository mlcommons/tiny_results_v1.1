#ifndef FPGACONVNET_H
#define FPGACONVNET_H

#include "dma_util.h"
#include "xaxidma.h"

typedef struct {

    // base address for
	UINTPTR base_addr;

    // DMA
    XAxiDma* dma_in;
    XAxiDma* dma_out;

    // register offsets
    unsigned int status_reg_offset;
	unsigned int size_in_offset;
	unsigned int size_out_offset;

} fpgaconvnet_t;

int fpgaconvnet_init(fpgaconvnet_t* dev,
        XAxiDma* dma_in, XAxiDma* dma_out,
        unsigned int base_addr);
void fpgaconvnet_reset(fpgaconvnet_t* dev);
void fpgaconvnet_set_size_in(fpgaconvnet_t* dev, int size);
void fpgaconvnet_set_size_out(fpgaconvnet_t* dev, int size);
void fpgaconvnet_transfer_featuremaps(fpgaconvnet_t* dev, u8* data_in,
        u8* data_out, unsigned int size_in, unsigned int size_out);

#endif
