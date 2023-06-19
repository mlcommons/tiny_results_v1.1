#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "xil_printf.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "xuartps.h"

#include "config.h"
#include "fpgaconvnet.h"

#include "api/submitter_implemented.h"
#include "api/internally_implemented.h"

#if EE_CFG_ENERGY_MODE == 1

//GPIO Config for Timestamp
#include "xgpiops.h" // GPIO PS Control Library
#define TIMER_PIN 12 //Arduino RST, Labeled "RST" on board between "IOREF" and "3.3V" on arduino shield connectors

//GPIO Config for Timestamp
XGpioPs Gpio;
XGpioPs_Config *GPIOConfigPtr;
int GPIO_Status;
#define TIMER_PIN

#endif

//#define UART_DEVICE_ID  XPAR_XUARTPS_0_DEVICE_ID
XUartPs Uart_Ps;		/* The instance of the UART Driver */

// hardware struct
fpgaconvnet_t runner;

// pointers for featuremaps
int16_t* featuremap_in;
int16_t* featuremap_out;
float* infer_results;

// create DMA instance
XAxiDma dma;

// timestamp information
XTime th_timer_start;
XTime th_timer_timestamp;
double th_calibration_time;
double th_timestamp_counts;

int16_t uint8_to_int9(uint8_t data, int offset) {
    return (int16_t) ( ((int16_t) data) + offset);
}

int16_t int8_to_int9(int8_t data, int offset) {
    return (int16_t) ( ((int16_t) data) + offset);
}

void softmax(float* data, size_t len) {
    /**
     * https://codereview.stackexchange.com/questions/180467/implementing-softmax-in-c
     */

    // find the max value
    float max = data[0];
    for(int i = 0; i < len; i++) {
        if(data[i] > max) {
            max = data[i];
        }
    }

    // sum of exponentials
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += expf(data[i] - max);
    }

    // perform softmax
    float offset = max + logf(sum+0.0000001);
    for (int i = 0; i < len; i++) {
        data[i] = expf(data[i] - offset);
    }

    return;
}

double get_elapsed_time(XTime start, XTime stop)
{
    return 1.0 * (stop - start) / (COUNTS_PER_SECOND);
}

void th_load_tensor() {

    // flush cache
    Xil_DCacheFlushRange((UINTPTR)featuremap_in, FEATUREMAP_IN_SIZE * sizeof(uint16_t));
    Xil_DCacheFlushRange((UINTPTR)featuremap_out, FEATUREMAP_OUT_SIZE * sizeof(uint16_t));

    // copy the buffer to the input pointer
#if BENCHMARK_INDEX == 4
    float* input = (float*) malloc(FEATUREMAP_IN_SIZE*sizeof(float));
    ee_get_buffer(input, FEATUREMAP_IN_SIZE * sizeof(float));
#else
    uint8_t* input = (uint8_t*) malloc(FEATUREMAP_IN_SIZE*sizeof(uint8_t));
    ee_get_buffer(input, FEATUREMAP_IN_SIZE * sizeof(uint8_t));
#endif

    // load and quantise  featuremap in channel-first order
    for (int i = 0; i < FEATUREMAP_IN_SIZE; i++) {
#if BENCHMARK_INDEX == 3
         featuremap_in[i] = int8_to_int9((int8_t) input[i], INPUT_OFFSET);
#elif BENCHMARK_INDEX == 4
         featuremap_in[i] = (int16_t)((input[i] / 0.3910152316093445) + 89);
#else
         featuremap_in[i] = uint8_to_int9(input[i], INPUT_OFFSET);
#endif
    }

    // free the input pointer memory
    free(input);

    return;

}

void th_infer() {

    // setup fpgaconvnet ip
    fpgaconvnet_reset(&runner);
    fpgaconvnet_set_size_out(&runner, FEATUREMAP_OUT_SIZE);

    // flush cache
    Xil_DCacheFlushRange((UINTPTR)featuremap_in, FEATUREMAP_IN_SIZE * sizeof(int16_t));
    Xil_DCacheFlushRange((UINTPTR)featuremap_out, FEATUREMAP_OUT_SIZE * sizeof(int16_t));
    Xil_DCacheFlushRange((UINTPTR)infer_results, FEATUREMAP_OUT_SIZE * sizeof(float));

    // perform featuremap transfer
    fpgaconvnet_transfer_featuremaps(&runner,
            (u8*) featuremap_in, (u8*) featuremap_out,
            FEATUREMAP_IN_SIZE * sizeof(int16_t),
            FEATUREMAP_OUT_SIZE * sizeof(int16_t));

    Xil_DCacheInvalidateRange((UINTPTR)featuremap_out, FEATUREMAP_OUT_SIZE * sizeof(int16_t));
    Xil_DCacheInvalidateRange((UINTPTR)infer_results, FEATUREMAP_OUT_SIZE * sizeof(float));

    // convert output to floating point
    for (int i = 0; i < FEATUREMAP_OUT_SIZE; i++) {
        infer_results[i] = (float) featuremap_out[i];
    }

    // perform softmax
#if BENCHMARK_INDEX != 4
     softmax(infer_results, FEATUREMAP_OUT_SIZE);
#endif

    return;

}

void th_results() {

#if BENCHMARK_INDEX == 4
    for (int i = 0; i < FEATUREMAP_OUT_SIZE; i++) {
        infer_results[i] = (infer_results[i] - (96)) * 0.36449846625328064;
    }
#endif

    th_printf("m-results-[");
    for (int i = 0; i < FEATUREMAP_OUT_SIZE; i++) {
        th_printf("%0.3f", infer_results[i]);
        if (i < (FEATUREMAP_OUT_SIZE - 1)) {
            th_printf(",");
        }
    }
    th_printf("]\r\n");

}


void th_final_initialize(void) {

	// initialise DMAs
	init_dma(&dma, DMA_DEVICE_ID);

    // initialise fpgaconvnet
    fpgaconvnet_init(&runner, &dma, &dma, RUNNER_BASEADDR);

    // allocate memory
    featuremap_in = (int16_t*) malloc(FEATUREMAP_IN_SIZE * sizeof(int16_t));
    featuremap_out = (int16_t*) malloc(FEATUREMAP_OUT_SIZE * sizeof(int16_t));
    infer_results = (float*) malloc(FEATUREMAP_OUT_SIZE * sizeof(float));

    return;

}

void th_timestamp(void) {
    unsigned long microSeconds = 0ul;
    /* USER CODE 2 BEGIN */
	XTime_GetTime(&th_timer_timestamp);
	th_timestamp_counts = get_elapsed_time(th_timer_start, th_timer_timestamp);
	microSeconds = th_timestamp_counts/th_calibration_time;
    /* USER CODE 2 END */
    /* This message must NOT be changed. */
    th_printf(EE_MSG_TIMESTAMP, microSeconds);
}

void th_pre() {}
void th_post() {}

// (from reference implementation)
void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// (from finn implementation)
void th_timestamp_initialize(void)
{

    /* USER CODE 1 BEGIN */
	XTime th_timer_stop;
	XTime_GetTime(&th_timer_start);
	usleep(1000); //sleep for 1000us to calibrate timer
	XTime_GetTime(&th_timer_stop);
	th_calibration_time = get_elapsed_time(th_timer_start, th_timer_stop)/1000;
    /* USER CODE 1 END */

    /* This message must NOT be changed. */
    th_printf(EE_MSG_TIMESTAMP_MODE);
    /* Always call the timestamp on initialize so that the open-drain output
       is set to "1" (so that we catch a falling edge) */
    th_timestamp();
}

// (from finn implementation)
void th_serialport_initialize(void) { }

// (from reference implementation)

char th_getchar() { return getchar(); }

int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

int th_vprintf(const char *format, va_list ap) {
    return vprintf(format, ap);
}

void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}
