#include "fpgaconvnet.h"
#include "dma_util.h"

int fpgaconvnet_init(fpgaconvnet_t* dev,
        XAxiDma* dma_in, XAxiDma* dma_out,
        unsigned int base_addr) {

    // initialise devices
    dev->base_addr = (UINTPTR) base_addr;
    dev->dma_in = dma_in;
    dev->dma_out = dma_out;

    // initialise status registers
    dev->status_reg_offset  = 0x0;
	dev->size_in_offset     = 0x4;
	dev->size_out_offset    = 0x8;

    return XST_SUCCESS;

}

void fpgaconvnet_reset(fpgaconvnet_t* dev) {
	Xil_Out32(dev->base_addr+dev->status_reg_offset, 0x2);
	Xil_Out32(dev->base_addr+dev->status_reg_offset, 0x0);
}

void fpgaconvnet_set_size_in(fpgaconvnet_t* dev, int size) {
	Xil_Out32(dev->base_addr + dev->size_in_offset, size);
}

void fpgaconvnet_set_size_out(fpgaconvnet_t* dev, int size) {
	Xil_Out32(dev->base_addr + dev->size_out_offset, size);
}

void fpgaconvnet_transfer_featuremaps(fpgaconvnet_t* dev, u8* data_in,
        u8* data_out, unsigned int size_in, unsigned int size_out) {

	// transfer featuremaps
	transfer_both(dev->dma_in, dev->dma_out, (UINTPTR*) data_in,
            (UINTPTR*) data_out, size_in, size_out);

}
