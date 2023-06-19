import os
import numpy as np
import cv2
import onnx
from onnx import numpy_helper
import onnxruntime as rt
from typing import OrderedDict


def inference_calibration_activations(data_path, graph, percentile=99.99, number=20):
    """inference to generate calibration activations
    """
    activations = graph.activations
    all_outputs = {}

    count_file = 0
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if count_file == number:
                break
            
            count_file += 1
            filepath = os.path.join(dirname, filename)
            img = preprocess_data(filepath, size = 96)
            output = inference_onnx(img, graph.onnx_model)
            for activation_name, activation_value in output:
                if activation_name in activations:
                    if activation_name in all_outputs:
                        all_outputs[activation_name] = np.concatenate((all_outputs[activation_name], activation_value.flatten()))
                    else:
                        all_outputs[activation_name] = activation_value.flatten()

    factors = {}
    bit_width = 8
    quantization_range = pow(2, bit_width) - 1
    factors["input"] = (quantization_range, 1, 0)  # set input layer range (0, quantization_range)

    for idx, (activation_name, activation_value) in enumerate(all_outputs.items()):
        activation_value = np.abs(activation_value)
        range_value = np.percentile(activation_value, percentile)
        factors[activation_name] = (get_quantizing_factor(range_value, "uint8"), range_value, idx+1)
    return factors

def read_data(image_path):
    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_data(image_path, size=300, mean=0):
    img = read_data(image_path)
    img = cv2.resize(img, (size, size))
    img = (img - mean)/ 255
    img = np.expand_dims(img.transpose((2, 0, 1)), axis = 0)
    return img

def get_quantizing_factor(range_a, format):

    # assert range_a>0, 'range of data must be positive'
    if isinstance(range_a, list):
        if format == 'int8': # -> [-128, 127] + 128
            return [127.0/np.float64(ra) for ra in range_a]
        elif format == 'uint8': # -> [0, 255]
            return [255.0/np.float64(ra) for ra in range_a]
        elif format == 'int7':
            return [63.0/np.float64(ra) for ra in range_a]
        else:
            raise NotImplementedError
    else:
        if format == 'int8': # -> [-128, 127] + 128
            return 127.0/np.float64(range_a)
        elif format == 'uint8': # -> [0, 255]
            return 255.0/np.float64(range_a)
        elif format == 'int7':
            return 63.0/np.float64(range_a)
        else:
            raise NotImplementedError
        
def onnx_infer_sess(onnx_model):
    model = onnx_model

    weights = OrderedDict()
    for init in model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    activations_out = []
    for node in model.graph.node:
        for output in node.output:
            activations_out.extend(node.output)
            model.graph.output.extend([onnx.ValueInfoProto(name = output)])
    # remove duplicates in activations_out(if have), with order
    seen = set()
    seen_add = seen.add
    activations_out = [x for x in activations_out if not (x in seen or seen_add(x))]

    sess = rt.InferenceSession(model.SerializeToString())
    label_name = activations_out
    return (sess, label_name)

def inference_onnx(data, onnx_model):
    """inference data using onnx model
    """
    sess, label_name = onnx_infer_sess(onnx_model)
    input_name = sess.get_inputs()[0].name
    
    out = sess.run(label_name, {input_name: data.astype(np.float32)})
    return zip(label_name, out)