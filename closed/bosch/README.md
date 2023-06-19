# Bosch Hardware-Aware Lowering Engine
The Bosch Hardware-Aware Lowering Engine (*HALE*) is a powerful commercial code generator designed specifically to 
generate optimized C code for microcontrollers.
Cortex-M-based architectures with and without DSP are in focus but other microcontroller architectures are also supported.
*HALE*'s generated code offers superior performance compared to TensorFlow Lite Micro while producing identical numerical results.
What sets *HALE* apart is its versatility, as it can optimize code for maximum performance, 
RAM usage, code size or a compromise of these quantities.
It even supports the deployment and optimization of multiple networks simultaneously.

# Run the benchmarks
In this submission we provide the code necessary to perform the MLPerf™ benchmarks and to show the API of *HALE*’s code.
We benchmarked the *HALE*-generated code on a Cortex-M0+, a Cortex-M4, two different Cortex-M7 and a Cortex-M33 board.
If you wish to verify our results, you can flash the corresponding board with the binary files that are included in our submission.
To flash the various boards from a Linux machine, simply connect the correct board to your machine and then do

cp <BINARY FILE> /media/$USER/DIS_F746NG

cp <BINARY FILE> /media/$USER/NOD_G0B1RE

cp <BINARY FILE> /media/$USER/NOD_H7A3ZIQ

cp <BINARY FILE> /media/$USER/NODE_L4R5ZI

cp <BINARY FILE> /media/$USER/NOD_U575ZI
