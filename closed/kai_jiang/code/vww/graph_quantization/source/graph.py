from re import A
import onnx
from onnx import numpy_helper
# from torch import F
import numpy as np
from typing import OrderedDict

from quantization import *
from config import *

class Conv:
    """Conv class
    """
    def __init__(self,) -> None:
        self.weight = None
        self.bias = None
        self.dilations = [1, 1]
        self.group = 1
        self.kernel_shape = None
        self.pads = [0, 0, 0, 0]
        self.strides = [1, 1]

        self.weight_q = None  # quantized weight
        self.bias_q = None
        self.activation_factor = None
        self.weight_factor = None
        self.deq_factor = None
        self.deq_factor_with_s3 = None


class Transpose:
    def __init__(self, perm) -> None:
        self.perm = perm

class Reshape:
    def __init__(self, shape) -> None:
        self.shape = shape

class Concat:
    def __init__(self, axis) -> None:
        self.axis = axis

class AveragePool:
    def __init__(self,) -> None:
        self.kernel_shape = None
        self.strides = None

class Softmax:
    def __init__(self, axis) -> None:
        self.axis = axis

class Matmul:
    def __init__(self, alpha, beta, transa, transb) -> None:
        self.alpha = alpha
        self.beta = beta
        self.transa = transa
        self.transb = transb
        self.weight = None
        self.bias = None

        self.weight_q = None  # quantized weight
        self.bias_q = None
        self.activation_factor = None
        self.weight_factor = None
        self.deq_factor = None
        self.deq_factor_with_s3 = None


class Node:
    """Node class
    """
    def __init__(self, node_id, name) -> None:
        self.node_id = node_id
        self.name = name
        self.node_type = None
        self.data = None
        self.activation_func_type = -1
        self.is_iterated = False
        self.weight_quantization_type = None
        self.last_quantization_node = False
        # self.has_add_child = False  # if has child with data type 'ADD'

        self.params_size = 0
        self.flops = 0
        self.memory_consumption = 0


class Graph:
    """Graph class
    """
    def __init__(self, nodes_num) -> None:
        self.nodes_num = nodes_num
        self.nodes = [None] * nodes_num
        self.node2outputs = None
        self.node2inputs = None
        self.input2nodes = None
        self.output2nodes = None
        self.activation_qutization_type = None
        self.activations_num = 0
        self.graph_inputs = None
        self.graph_input_shape = []
        self.graph_outputs = None
        self.activations = None  # activation name
        self.activations_shape = {}  #TODO: using activation class
        self.onnx_model = None  # onnx model, used onnx runtime to inference


def extract_onnx(model_file):
    onnx_model = onnx.load(model_file)
    onnx_graph = onnx_model.graph

    sess, _ = onnx_infer_sess(onnx_model)
    onnx_outputs = [x.name for x in sess.get_outputs()]
    ort_outputs = OrderedDict(zip(onnx_outputs, sess.get_outputs()))

    activations = []
    node2outputs = OrderedDict()  # key:node_name, value: outputs of the node
    node2inputs = OrderedDict()  # key:node_name, value: inputs of the node
    input2nodes = OrderedDict()  # key:input, value: nodes(node_name) that take this input as the input
    output2nodes = OrderedDict()  # key:output, value: nodes(node_name) that take this output as the output

    activations_out = []

    # graph parameters
    weights = OrderedDict()
    for init in onnx_graph.initializer:
        # print(init.name)
        weights[init.name] = numpy_helper.to_array(init)

    node_num = 0
    for node in onnx_graph.node:
        # TODO: trancate the graph for Mobilenetv2-ssd, just because of unsupported operations, need to add such operations later
        node_num += 1
        if node.name == END_NODE:
            break
    
    graph = Graph(node_num)
    graph.onnx_model = onnx_model

    # get graph input shape, assuming input[0] is the real input
    for idx, dv in enumerate(onnx_graph.input[0].type.tensor_type.shape.dim):
        if idx == 0:
            graph.graph_input_shape = [dv.dim_value]
        else:
            graph.graph_input_shape.append(dv.dim_value)

    
    for node_id, onnx_node in enumerate(onnx_graph.node):
        node_name = onnx_node.name
        node = Node(node_id, node_name)

        # add op_type, weight, attribute
        data = None
        if onnx_node.op_type == 'Conv':
            if onnx_node.attribute[1].i == 1:  # group
                op_type = Operations['conv']  # conv2d
            else:
                op_type = Operations['group_conv']  # depth_conv2d
            
            data = Conv()
            for weight_name in onnx_node.input:
                if weight_name in weights:
                    weight = weights[weight_name]
                    # print(weight_name, weight.shape)
                    if len(weight.shape) == 4:
                        data.weight = weight
                    elif len(weight.shape) == 1:
                        data.bias = weight
            
            data.dilations = onnx_node.attribute[0].ints
            data.group = onnx_node.attribute[1].i
            data.kernel_shape = onnx_node.attribute[2].ints
            data_pad = onnx_node.attribute[3].ints
            # assert data_pad[0] == data_pad[1] and data_pad[2] == data_pad[3], "wrong pad!"
            data.pads = data_pad
            data.strides = onnx_node.attribute[4].ints

        elif onnx_node.op_type == 'Clip':
            if onnx_node.attribute[1].f == 0.0 and onnx_node.attribute[0].f == 6.0:
                op_type = Operations['relu6']  # relu6
            else:
                op_type = 100  # clip
        
        elif onnx_node.op_type == 'Relu':
            op_type = Operations['relu']
        
        elif onnx_node.op_type == 'Add':
            op_type = Operations['add']

        elif onnx_node.op_type == 'Transpose':
            data = Transpose(onnx_node.attribute[0].ints)
            op_type = Operations['transpose']

        elif onnx_node.op_type == 'Reshape':
            for weight_name in onnx_node.input:
                if weight_name in weights:
                    weight = weights[weight_name]
                    data = Reshape(weight)
                    break
            op_type = Operations['reshape']
        
        elif onnx_node.op_type == 'Concat':
            data = Concat(onnx_node.attribute[0].i)
            op_type = Operations['concat']
        
        elif onnx_node.op_type == 'AveragePool':
            data = AveragePool()
            data.kernel_shape = onnx_node.attribute[1].ints
            data.strides = onnx_node.attribute[2].ints
            op_type = Operations['average_pool']
        
        elif onnx_node.op_type == 'Softmax':
            data = Concat(onnx_node.attribute[0].i)
            op_type = Operations['softmax']
        
        elif onnx_node.op_type == 'Gemm':
            data = Matmul(onnx_node.attribute[0].f, onnx_node.attribute[1].f, onnx_node.attribute[2].i, onnx_node.attribute[3].i)
            for weight_name in onnx_node.input:
                if weight_name in weights:
                    weight = weights[weight_name]
                    if len(weight.shape) == 2:
                        data.weight = weight
                    elif len(weight.shape) == 1:
                        data.bias = weight
            op_type = Operations['matmul']
        
        else:
            raise ValueError("unsupported op_type:{}!".format(onnx_node.op_type))

        node.data = data
        node.node_type = op_type
        graph.nodes[node_id] = node
        
        # add input to activation
        for input in onnx_node.input:
            if not input in weights:
                # update input2nodes
                if input in input2nodes:
                    input2nodes[input].append(node_name)
                else:
                    input2nodes[input] = [node_name]
                
                # update node2inputs
                if node_name in node2inputs:
                    node2inputs[node_name].append(input)
                else:
                    node2inputs[node_name] = [input]

        # # add output to activation
        for node_output_name in onnx_node.output:
            if not node_output_name in weights:
                if node_output_name not in activations_out:
                    activations_out.append(node_output_name)
                # update node2outputs
                if node_name in node2outputs:
                    node2outputs[node_name].append(node_output_name)
                else:
                    node2outputs[node_name] = [node_output_name]
                
                # update output2nodes
                if node_output_name in output2nodes:
                    output2nodes[node_output_name].append(node_name)
                else:
                    output2nodes[node_output_name] = [node_name]
        
        
        if node_name == END_NODE:
            break
        
    # update graph input, output
    graph_input_name = [x.name for x in onnx_graph.input]
    activations = graph_input_name + activations_out

    # update activations_shape
    for activ_name in activations_out:
        activ_shape = ort_outputs[activ_name].shape
        graph.activations_shape[activ_name] = activ_shape
    
    nodes_widefirst_travel = [graph.nodes[0].name]
    accessed_activations = []
    deepfirst_to_widefirst(GRAPH_OUTPUT_NAME, nodes_widefirst_travel, accessed_activations, node2outputs, input2nodes, node2inputs, graph.nodes[0].name)  # reorder nodes by widefirst_travel
    # reorder nodes based on nodes_widefirst_travel
    widefirst_nodes = []
    for idx, node_name in enumerate(nodes_widefirst_travel):
        node_id = get_nodeid_by_name(graph.nodes, node_name)
        node = graph.nodes[node_id]
        node.node_id = idx
        widefirst_nodes.append(node)
    graph.nodes = widefirst_nodes

    # convert node2outputs, input2nodes, node2inputs, output2nodes represention from name to id
    node2output_ids = OrderedDict()
    for node_name, output_names in node2outputs.items():
        node_id = get_nodeid_by_name(graph.nodes, node_name)
        for output_name in output_names:
            output_id = activations.index(output_name)
            if node_id in node2output_ids:
                node2output_ids[node_id].append(output_id)
            else:
                node2output_ids[node_id] = [output_id]
    
    node2input_ids = OrderedDict()
    for node_name, input_names in node2inputs.items():
        node_id = get_nodeid_by_name(graph.nodes, node_name)
        for input_name in input_names:
            input_id = activations.index(input_name)
            if node_id in node2input_ids:
                node2input_ids[node_id].append(input_id)
            else:
                node2input_ids[node_id] = [input_id]
    
    input_id2nodes = OrderedDict()
    for input_name, node_names in input2nodes.items():
        input_id = activations.index(input_name)
        for node_name in node_names:
            node_id = get_nodeid_by_name(graph.nodes, node_name)
            if input_id in input_id2nodes:
                input_id2nodes[input_id].append(node_id)
            else:
                input_id2nodes[input_id] = [node_id]
    
    output_id2nodes = OrderedDict()
    for output_name, node_names in output2nodes.items():
        output_id = activations.index(output_name)
        for node_name in node_names:
            node_id = get_nodeid_by_name(graph.nodes, node_name)
            if output_id in output_id2nodes:
                output_id2nodes[output_id].append(node_id)
            else:
                output_id2nodes[output_id] = [node_id]

    graph.node2outputs = node2output_ids
    graph.input2nodes = input_id2nodes
    graph.node2inputs = node2input_ids
    graph.output2nodes = output_id2nodes
    graph.activations = activations
    graph.activations_num = len(activations)
    graph.graph_inputs = graph_input_name
    graph.graph_outputs = GRAPH_OUTPUT_NAME
    return graph


def deepfirst_to_widefirst(graph_output, widefirst_nodes, accessed_activations, node2outputs, input2nodes, node2inputs, node_name):
    """convert nodes list with deepfirst order to list with widefirst order
    """ 
    # node = deepfirst_nodes[node_id]
    node_outputs = node2outputs[node_name]
    for output in node_outputs:
        accessed_activations.append(output)
        if output in graph_output:
            break
        next_nodes = input2nodes[output]
        
        computable_next_nodes = []  # computable nodes, because inputs of them are all in accessed_activations
        for next_node in next_nodes:
            next_node_inputs = node2inputs[next_node]
            flag_computable = True
            for next_node_input in next_node_inputs:
                if next_node_input not in accessed_activations:
                    flag_computable = False
                    break
                # else:
                #     widefirst_nodes.append(next_node)
            if flag_computable:
                computable_next_nodes.append(next_node)
        
        widefirst_nodes.extend(computable_next_nodes)
        
        for next_node in computable_next_nodes:
            deepfirst_to_widefirst(graph_output, widefirst_nodes, accessed_activations, node2outputs, input2nodes, node2inputs, next_node)

# TODO: use hash map
def get_nodeid_by_name(nodes, node_name):
    for node in nodes:
        if node.name == node_name:
            return node.node_id
    raise ValueError("wrong node name")


def fuse_activations(graph):
    """fuse specific nodes(convolution) and its activation function
    """
    graph = fuse_nodes(graph, Operations['conv'], Operations['relu6'])
    graph = fuse_nodes(graph, Operations['group_conv'], Operations['relu6'])
    graph = fuse_nodes(graph, Operations['conv'], Operations['relu'])
    graph = fuse_nodes(graph, Operations['group_conv'], Operations['relu'])
    return graph


def fuse_nodes(graph, type1, type2):
    if (type1 != Operations['conv'] and type1 != Operations['group_conv']) or \
        (type2 != Operations['relu'] and type2 != Operations['relu6']):
        raise NotImplementedError

    for node in graph.nodes:
        if node.node_type == type1:
            node_output = graph.node2outputs[node.node_id]
            if len(node_output) == 1:
                next_nodes_id = graph.input2nodes[node_output[0]]
                if len(next_nodes_id) == 1 and graph.nodes[next_nodes_id[0]].node_type == type2:
                    node.activation_func_type = type2
                    graph = delete_node(graph, next_nodes_id[0])
    return graph


def delete_node(graph, node_id):
    """delete node in the graph
    """
    nodes = graph.nodes
    node2outputs = graph.node2outputs
    node2inputs = graph.node2inputs
    # input2nodes = graph.input2nodes
    output2nodes = graph.output2nodes
    activations = graph.activations

    new_nodes = []
    new_node2outputs = OrderedDict()
    new_node2inputs = OrderedDict()

    node_inputs = node2inputs[node_id]
    if len(node_inputs) != 1:
        raise ValueError("only node with one input can be deleted!")
    if activations[node_inputs[0]] in graph.graph_inputs:  # delete node with graph input
        raise NotImplementedError
    new_activations = activations[:node_inputs[0]] + activations[(node_inputs[0] + 1):]
    
    father_node_ids = output2nodes[node_inputs[0]]

    for node1 in nodes:
        node_id1 = node1.node_id
        if node_id1 < node_id:
            new_nodes.append(node1)
            new_node2inputs[node_id1] = [(new_activations.index(activations[act])) for act in node2inputs[node_id1]]
            if node_id1 in father_node_ids:
                new_node2outputs[node_id1] = [(new_activations.index(activations[act])) for act in node2outputs[node_id]]
            else:
                new_node2outputs[node_id1] = [(new_activations.index(activations[act])) for act in node2outputs[node_id1]]
        elif node_id1 > node_id:
            new_node_id1 = node_id1 - 1
            node1.node_id = new_node_id1
            new_nodes.append(node1)
            new_node2outputs[new_node_id1] = [(new_activations.index(activations[act])) for act in node2outputs[node_id1]]
            new_node2inputs[new_node_id1] = [(new_activations.index(activations[act])) for act in node2inputs[node_id1]]
        
    graph.nodes_num -= 1
    graph.nodes = new_nodes
    graph.node2outputs = new_node2outputs
    graph.node2inputs = new_node2inputs
    graph.activations = new_activations
    graph.activations_num -= 1

    graph = generate_activation_to_node(graph)
    return graph


def generate_activation_to_node(graph):
    """generate input2nodes, output2nodes based on node2inputs, node2outputs
    """
    node2inputs = graph.node2inputs
    node2outputs = graph.node2outputs
    activations_num = graph.activations_num

    input2nodes = OrderedDict()
    output2nodes = OrderedDict()

    for idx in range(activations_num - len(graph.graph_outputs)):
        input2nodes[idx] = []
    for idx in range(len(graph.graph_inputs), activations_num + 1):
        output2nodes[idx] = []

    for node, inputs in node2inputs.items():
        for inp in inputs:
            input2nodes[inp].append(node)
    
    for node, outputs in node2outputs.items():
        for outp in outputs:
            output2nodes[outp].append(node)
    
    graph.input2nodes = input2nodes
    graph.output2nodes = output2nodes
    return graph


def extract_onnx_weights(model_file, output_file, save=False):
    """extract onnx model weights and save to pkl file
    """
    model = onnx.load(model_file)
    onnx_graph = model.graph

    # graph parameters
    weights = OrderedDict()
    for init in onnx_graph.initializer:
        # print(init.name)
        weights[init.name] = numpy_helper.to_array(init)
    if save:
        with open(output_file, "wb") as f:
            pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)


# def set_has_add_child(graph):
#     node2outputs = graph.node2outputs
#     input2nodes = graph.input2nodes
#     nodes = graph.nodes
#     activations_out = graph.activations_out
#     graph_outputs = graph.graph_outputs
#     graph_inputs_num = len(graph.graph_inputs)
    
#     for node in nodes:
#         node_output = node2outputs[node.node_id]
#         if len(node_output) > 1:
#             raise NotImplementedError("only support nodes with one output!")
#         if activations_out[node_output[0] - graph_inputs_num] in graph_outputs:
#             break
#         child_nodes = input2nodes[node_output[0]]
#         for child_node in child_nodes:
#             if nodes[child_node].node_type == Operations['add']:
#                 graph.nodes[node.node_id].has_add_child = True
#     return graph
    
    