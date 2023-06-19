#!/bin/bash

## download the tflite model
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite

## convert to ONNX
python -m tf2onnx.convert --tflite vww_96_int8.tflite --output model_quant.onnx  --inputs-as-nchw "input_1_int8"

