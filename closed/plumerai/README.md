# Plumerai Inference Engine for microcontrollers

[Plumerai](https://plumerai.com/) is a company making deep learning tiny.
Plumerai’s inference engine tool and runtime provides [state-of-the-art](https://blog.plumerai.com/2021/10/cortex-m-inference-software/) fast execution of deep learning models in tiny memory spaces.
Compared to TensorFlow Lite for Microcontrollers with Arm’s CMSIS-NN kernels, [models run on faster, with less RAM, and with a smaller code size](https://blog.plumerai.com/2022/11/mlperf-tiny-1.0/).
This enables AI workloads on tiny microcontrollers.
It reads in 8-bit TensorFlow Lite files (.tflite) and generates a C and C++ library that includes the model, runtime, and optimized kernels.

The results of this submission are based on that software: the Plumerai Inference Engine.
The Plumerai Inference Engine is commercially available to be used in the cloud or on developer's machines for Linux, Windows, and macOS.
It is optimized for Arm Cortex-M, ESP32-S3, ARC EM4, and RISC-V microcontrollers.

Want to see how fast your models can run? You can submit them for free on our [Plumerai Benchmark service](https://plumerai.com/benchmark).
We email you the results in minutes.

The Plumerai Inference Engine does no pruning, quantization, or binarization.
Model accuracy stays the same, but compared to other inference engines inference speed goes up, memory usage goes down, and code size is reduced.

For more information, see the [online Plumerai Inference Engine documentation](https://docs.plumerai.com/latest/inference_engine/).


## The systems under test

We ran the Plumerai Inference Engine on the following microcontrollers:
* STMicroelectronics STM32L4R5 (`NUCLEO_L4R5ZI`) with Cortex-M4
* STMicroelectronics STM32U575 (`NUCLEO_U575ZI_Q`) with Cortex-M33
* STMicroelectronics STM32H7A3 (`NUCLEO_H7A3ZI_Q`) with Cortex-M7
* STMicroelectronics STM32F746 (`DISCO-F746NG`) with Cortex-M7
* Infineon Cypress CY8CPROTO-062-4343W (`cy8cproto_062_4343w`) with Cortex-M4

More details about the devices can be found under `systems/<device_name>.json`.

Note that the same code runs on many other devices. The example code submitted here is based on [MBED](https://os.mbed.com/), meaning only microcontrollers with MBED support will work.
However, the Plumerai Inference Engine does not have this limitation and can work on other microcontrollers as well.


## The code

See [code/README.md](code/README.md) for details.


## The results

The logs for accuracy and performance can be found in the `results` subdirectory grouped by device and benchmark.
They are the (unmodified) outputs of the official [EEMBC runner](https://github.com/eembc/energyrunner/) for accuracy and performance measurements.
In summary the results are:

Latency results in ms:

| Board               | SoC             | MCU            |  Clock |     KWS |     IC |    VWW |   AD |
|---------------------|-----------------|----------------|-------:|--------:|-------:|-------:|-----:|
| NUCLEO-L4R5ZI       | STM32L4R5ZIT6PU | Arm Cortex-M4  | 120MHz |   47.78 | 167.76 |  98.76 | 4.05 |
| NUCLEO-U575ZI-Q     | STM32U575ZIT6QU | Arm Cortex-M33 | 160MHz |   30.04 | 104.29 |  59.49 | 3.67 |
| NUCLEO-H7A3ZI-Q     | STM32H7A3ZIT6QU | Arm Cortex-M7  | 280MHz |   13.86 |  50.93 |  26.82 | 1.35 |
| DISCO-F746NG        | STM32F746NGH6U  | Arm Cortex-M7  | 216MHz |   17.75 |  62.88 |  34.08 | 1.71 |
| CY8CPROTO_062_4343W | PSoC 62         | Arm Cortex-M4  | 150MHz |   54.06 | 188.96 | 108.30 | 4.46 |

Accuracy results (the same for each board):

|     |     accuracy |
|-----|--------------|
| VWW | Top-1: 84.9% |
| IC  | Top-1: 88.0% |
| KWS | Top-1: 90.2% |
| AD  |    AUC: 0.86 |
