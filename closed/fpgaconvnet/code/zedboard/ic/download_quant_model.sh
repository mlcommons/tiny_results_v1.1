#!/bin/bash

## download the tflite model
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite

## convert to ONNX
python -m tf2onnx.convert --tflite pretrainedResnet_quant.tflite --output model_quant.onnx  --inputs-as-nchw "input_1_int8"

