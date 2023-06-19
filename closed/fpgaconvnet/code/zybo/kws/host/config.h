#ifndef CONFIG_H_
#define CONFIG_H_

#include "xparameters.h"

/**
 * Define the mode for running
 */
#define EE_CFG_ENERGY_MODE 0

/**
 * Define the base address for the hardware
 */
#define RUNNER_BASEADDR XPAR_PARTITION_0_BASEADDR
#define DMA_DEVICE_ID   XPAR_DMA_DEVICE_ID

/**
 * Define the benchmark
 * 1 = VWW
 * 2 = IC
 * 3 = KWS
 * 4 = AD
 */
#define BENCHMARK_INDEX 3

/**
 * Define the name of the platform
 */
#define PLATFORM_NAME "fpgaconvnet-zybo-int8-dense"

/**
 * Define the offset applied to the input featuremap
 */
#define INPUT_OFFSET 0

#endif
