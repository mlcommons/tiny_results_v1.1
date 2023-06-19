if [[ -f mbedignore ]]; then
  mv mbedignore .mbedignore
fi
if [[ ! -f .mbed ]]; then
  mbed new --create-only .
fi
mbed deploy
cp ../../arm_convolve_HWC_q7_fast_nonsquare.fixed CMSIS_5/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c
touch onnc_runtime.h
mbed compile --source . -m NUCLEO_L4R5ZI -t GCC_ARM --build BUILD --profile release.json
