from typing import OrderedDict
import numpy as np
import pickle
import torch
import cv2
import struct
import onnx
from onnx import numpy_helper
import onnxruntime as rt

from config import *


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


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


def save_list_to_text(data, file_name, interval=100, ishex=False, isfloat=False):
    """
    save data to file_name
    """
    data_num = len(data)
    with open(file_name, 'w') as f:
        for idx, item in enumerate(data):
            if ishex:
                if type(data[0]) == np.float32 or type(data[0]) == float:
                    item = float_to_hex(item)
                elif type(data[0]) == np.int32 or type(data[0]) == np.int8 or type(data[0]) == np.int16 or type(data[0]) == int or type(data[0]) == np.uint8:
                    item = hex(item)
                else:
                    raise ValueError("unsopperted data type:{}!".format(type(data[0])))
            if isfloat:
                if (idx > 0) and (idx % interval == 0):
                    f.write(("%.6f,\n" % item))
                elif idx == data_num - 1:
                    f.write(("%.6f," % item))
                else:
                    f.write(("%.6f, " % item))
            else:
                if (idx > 0) and (idx % interval == 0):
                    f.write(("%s,\n" % item))
                elif idx == data_num - 1:
                    f.write(("%s," % item))
                else:
                    f.write(("%s, " % item))

def save_twodimension_data(file_name, data):
    with open(file_name, 'w') as file:
        file.write('[')
        for idx, row in enumerate(data):
            if idx == len(data) - 1:
                file.write(str(row) + ',')
            else:
                file.write(str(row) + ', ')
        file.write(']')


def load_pickle_outputs(filepath):
    """load pickle files"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_shift_n(value):
    """
    shift floated value to int32, return shift number and int32 value
    value: floated value
    """
    assert pow(2, 31) > value >= -pow(2, 31), " value {} is out of int32 range ".format(value)
    sign = 1
    n = 0
    if value < 0:
        # convert to positive
        sign = -1
        value = value * sign

    if value == 0:
        return [0, 0]
    elif value >= 1:
        while value >= 1 and n < 32:
            value = value / 2
            n = n + 1
    else:
        while value < 0.5 and n > -32:
            value = value * 2
            n = n - 1
    shift_n = 31 - n

    shifted_value = value * sign
    i = 0
    while i < 31:
        shifted_value = shifted_value * 2
        i = i + 1

    return [np.int(shifted_value), shift_n]


def read_image(image_file, input_shape, format="uint8"):
    assert len(input_shape) == 4, "input shape should be nhwc"
    img = preprocess_data(image_file, size=input_shape[2])
    # img_dict = {}
    # img_dict["input"] = img
    # with open("temps/COCO_train2014_000000000036.pkl", "wb") as f:
    #     pickle.dump(img_dict, f, pickle.HIGHEST_PROTOCOL)
    img = img.flatten()
    if format == "uint8":
        img = np.uint8(np.clip(np.rint(img * 255), 0, 255)) # original img value range from 0 to 1, convert to (0, 255)
    else:
        img = np.int8(np.clip(np.rint(img * 127), -127, 127))  # convert to (0, 127)
    return img
    # print(len(img))
    # save_list_to_text(img.astype(np.int32), save_file, ishex=True)


def compare_two_list(file1, file2):
    """compare contents of file1 and file2
    """
    data1 = []
    data2 = []
    with open(file1, 'r') as f1:
        for line in f1:
            line_list = [elem.strip(',') for elem in line.strip().split(', ')]  # python
            # line_list = [elem.strip(',') for elem in line.strip().split(' ')]
            data1.extend(line_list)
    with open(file2, 'r') as f2:
        for line in f2:
            line_list = [elem.strip(',') for elem in line.strip().split(' ')]   # c
            data2.extend(line_list)
    len_data1 = len(data1)
    len_data2 = len(data2)
    data1 = [float(elem) for elem in data1]
    data2 = [float(elem) for elem in data2]
    assert (len_data1 == len_data2), "unequal lengths!"
    diff = []
    for elem1, elem2 in zip(data1, data2):
        diff.append(abs(elem1 - elem2))
    print("max diff:{}".format(max(diff)))
    print(sum(diff), np.abs(data1).sum(), sum(diff) / len(diff), sum(diff) / np.abs(data1).sum())
    diff_np = np.array(diff)
    # print(diff_np[diff_np>0])
    
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


def get_feature_map(img_file, graph, save_path, activation_id=None, save=False):
    """inference and save feature maps 
    """
    img = preprocess_data(img_file, size=96)
    output = inference_onnx(img, graph.onnx_model)
    activations = graph.activations
    output2nodes = graph.output2nodes
    if activation_id:
        file_name = save_path + str(activation_id) + ".txt"
        activation_name = activations[activation_id]
        for name, value in output:
            if activation_name == name:
                print(value)
                if save:
                    save_list_to_text(value, file_name, isfloat=True)
                break
    else:
        if save:
            for name, value in output:
                if name in activations:
                    activation_id = activations.index(name)
                    node_id = output2nodes[activation_id]
                    if activation_id == 0:
                        activation_id = 0
                    file_name = save_path + str(node_id[0]) + ".txt"
                    save_list_to_text(value.flatten(), file_name, isfloat=True)


def compute_percentile(data, percentile):
    data = np.abs(data)
    if percentile == 1.0:
        upper = 1000
        percent_value = np.max(data[data < upper])
    else:
        data = np.sort(data)
        elem_num = data.size
        index = np.int32(np.round(elem_num * percentile) - 1)
        percent_value = data[index]
    
    # a verry small value of range will cause a very large value of quantization_factor, and causing int32 overflow
    # just use a normal value such as 1, this won't cause side effect
    if percent_value < 0.0000001:
        percent_value = 1  
    return percent_value


def convert_onnx(predictor, net, img_file):
    img = preprocess_data(img_file, size=96)
    img = predictor.transform(img)
    img = img.unsqueeze(0)
    img = img.to(predictor.device)
    dummy_input = img
    dummy_output = net(dummy_input) 
    model_trace = torch.jit.trace(net, dummy_input)
    torch.onnx.export(model_trace, dummy_input, f'v2_trace.onnx', example_outputs=dummy_output)