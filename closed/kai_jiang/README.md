# MLPerf Tiny benchmark v1 - Closed Division

This document provides an overview of our submission to the MLPerf Tiny benchmark v1.1. The benchmarks were recorded on the ZCU106. The submission contains performance results. Our solution is as below：

## convert tflite model to onnx model 

Convert vww_96_float.tflite to vww_96_float.onnx using [tflite2onnx]https://github.com/zhenhuaw-me/tflite2onnx.

## Generate quantized model data

Use our model parsing tool for model parsing, model quantization, and model serialization to generate quantized model data files. The model parsing tool is placed in the ```code/vww/graph_quantization``` folder. These model data files are placed in the ```code/vww/graph_quantization/temps/model_data``` folder, including model architecture, weight data, quantization information, and other data.

## Generate ANPU hardware execution control instructions

Based on the model architecture data, we dissect the model manually and generate ANPU hardware execution control instructions, which is located at ```code/vww/hardware/Hardware_Exucution_Control_Instruction_for_ANPU.xlsx```. 

## Generate model data used for ANPU

Use scripts to rearrange weight data, quantization data, and quantization offset data in the model data files to generate weight data, quantization data, and quantization offset data used for ANPU. These scripts are placed in the ```code/vww/hardware/convert_to_anpu_model_data``` folder, and their outputs are placed in the ```code/vww/hardware/convert_to_anpu_model_data/anpu_model_data``` folder.

## Integrate into an executable software program

Integrate ANPU's hardware execution control instructions, weight data, quantization data, and quantization offset data into an executable software program.

## Prepare for testing

1. Power on the ZCU106 board; use a JTAG downloader to write the "SoC+ANPU" BIT file, which is located at ```code/vww/hardware/hw.bit```. Then check if the SoC is working properly using CDK IDE, and flash the executable software program onto the SoC. Resources occupied by hardware_design is shown in ```code/vww/hardware/Resources_Occupied_By_Hardware_Design.png```
2. The executable software program sends ANPU's hardware execution control instructions, weight data, quantization data, and quantization offset data to ANPU.
3. The host reads an image from the dataset and sends it to the SoC via a serial port. When the SoC receives it, it sends the image to ANPU. ANPU caches the image and all intermediate calculation results in the accelerator memory and sends the calculation results to the SoC after the computation is done.
4. The SoC performs data post-processing and sends it to the host via a serial port, completing the computation process for a single image.
