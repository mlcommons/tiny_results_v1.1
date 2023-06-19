#include "dma_util.h"

int init_dma(XAxiDma* dma, int DMA_ID)
{
    int Status;

    XAxiDma_Config* config = XAxiDma_LookupConfig(DMA_ID);
    if (!config) {
        xil_printf("No config found for %d\r\n", DMA_ID);
        return XST_FAILURE;
    }
    Status = XAxiDma_CfgInitialize(dma, config);
    if (Status != XST_SUCCESS) {
        xil_printf("DMA initialisation  failed %d\r\n", Status);
        return XST_FAILURE;
    }
    if(XAxiDma_HasSg(dma)){
        xil_printf("DMA configured for SG mode \r\n");
        return XST_FAILURE;
    }
    Status= XAxiDma_Selftest(dma);
    if (Status != XST_SUCCESS) {
        xil_printf("DMA self-test failed %d\r\n", Status);
        return XST_FAILURE;
    }
    //xil_printf("DMA %d initialised\n\r", DMA_ID);

    XAxiDma_IntrDisable(dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    return XST_SUCCESS;
}

int transfer_in(XAxiDma* dma_in, UINTPTR* in, u32 size) {

    // flush cache
    Xil_DCacheFlushRange((UINTPTR)in , size);

    int status = XAxiDma_SimpleTransfer(dma_in, (UINTPTR)in, size, XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) {
        xil_printf("\n\rERROR in\tDMA_DataTransfer()\tDEV_TO_DMA\n\r");
        return XST_FAILURE;
    }

    // wait for dma in to stop being busy
    while( XAxiDma_Busy(dma_in, XAXIDMA_DMA_TO_DEVICE) );

    return XST_SUCCESS;
}

int transfer_both(XAxiDma* dma_in, XAxiDma* dma_out, UINTPTR* in, UINTPTR* out, u32 size_in, u32 size_out)
{

    int status;

    status = XAxiDma_SimpleTransfer(dma_in, (UINTPTR)in, size_in, XAXIDMA_DMA_TO_DEVICE);
    if (status != XST_SUCCESS) {
        xil_printf("\n\rERROR in\tDMA_DataTransfer()\tDEV_TO_DMA\n\r");
        return XST_FAILURE;
    }

    // start transfer
    status = XAxiDma_SimpleTransfer(dma_out, (UINTPTR)out, size_out, XAXIDMA_DEVICE_TO_DMA);
    if (status != XST_SUCCESS) {
        xil_printf("\n\rERROR in\tDMA_DataTransfer()\tDEV_TO_DMA\n\r");
        return XST_FAILURE;
    }

    // wait for transfer to finish
    int dma_in_busy = 1;
    int dma_out_busy = 1;
    do
    {
        dma_in_busy = XAxiDma_Busy(dma_in, XAXIDMA_DMA_TO_DEVICE);
        dma_out_busy = XAxiDma_Busy(dma_out, XAXIDMA_DEVICE_TO_DMA);
    }
    while(dma_in_busy|dma_out_busy);

    return XST_SUCCESS;
}
