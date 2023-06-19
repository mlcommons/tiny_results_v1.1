#!/bin/bash

## download the tflite model
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite

## convert to ONNX
python -m tf2onnx.convert --tflite kws_ref_model.tflite --output model_quant.onnx  --inputs-as-nchw "input_1"


