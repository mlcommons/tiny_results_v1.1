#include <stdio.h>
#include "xil_printf.h"

#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"

int main()
{
    ee_benchmark_initialize();
    while (1) {
        char c = inbyte();
        if (c == '\000') continue;
        if (c == '\r') continue; //xSDK Serial Terminal Fix
        ee_serial_callback(c);
    }
    return 0;
}
