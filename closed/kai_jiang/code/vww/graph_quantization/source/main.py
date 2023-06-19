import argparse

from quantization import *
from utils import *
from graph import *
from serialize import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='models/vww_96_float.onnx', 
                        help='model file path')
    parser.add_argument('-c', '--calibration_path', type=str, default='../../data/vww/calibration_official', 
                        help='calibration data path')
    parser.add_argument('-o', '--output_model_path', type=str, default='temps/model_data/', 
                        help='output model data path')

    args = parser.parse_args()
    onnx_file = args.model_path
    calibration_data_path = args.calibration_path
    prefix = args.output_model_path

    graph = extract_onnx(onnx_file)
    activation_factors = inference_calibration_activations(calibration_data_path, graph, mode="percentile", number=20)

    graph = fuse_activations(graph)
    graph = set_quantization_settings(graph)
    graph = quantize_weights(graph, activation_factors)
    md = ModelData(graph)
    md.create_txt(prefix)
