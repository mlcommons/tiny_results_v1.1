import numpy as np
import pickle
import os
from tqdm import tqdm
import copy
from scipy import stats

from utils import *
import Q

def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value. 
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin

        if num_merged_bins >= 10:
            num_merged_bins = 10
        
        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


def inference_calibration_activations(data_path, graph, mode="kl",
                                        percentile=99.99, number=20):
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
    bit_width = 0
    if ActivationQuantizationType == 'uint8':
        bit_width = 8
    elif ActivationQuantizationType == 'int8':  # int8
        bit_width = 7
    elif ActivationQuantizationType == 'int7':  # int7
        bit_width = 6
    else:
        raise NotImplementedError
    quantization_range = pow(2, bit_width) - 1
    factors["input"] = (quantization_range, 1, 0)  # set input layer range (0, quantization_range)

    if mode == "percentile":
        assert 0 <= percentile <= 100, "invalid percentile value"
        for idx, (activation_name, activation_value) in enumerate(all_outputs.items()):
            activation_value = np.abs(activation_value)
            range_value = np.percentile(activation_value, percentile)
            factors[activation_name] = (Q.get_quantizing_factor(range_value, ActivationQuantizationType), range_value, idx+1)
    elif mode == "kl":
        for idx, (activation_name, activation_value) in enumerate(all_outputs.items()):
            acti_value = np.abs(activation_value)
            acti_max = np.max(acti_value)
            distribution_interval = acti_max / 2048.0
            acti_distribution, hist_edge = np.histogram(acti_value, bins=2048, range=(0, acti_max))
            threshold_bin = threshold_distribution(acti_distribution, quantization_range + 1)
            range_value = (threshold_bin + 0.5) * distribution_interval
            factors[activation_name] = (Q.get_quantizing_factor(range_value, ActivationQuantizationType), range_value, idx+1)
    else:
        raise NotImplementedError("cannot support algorithm {}", mode)

    return factors
    

def set_quantization_settings(graph):
    """set quantization settings for graph nodes
    """
    graph.activation_qutization_type = ActivationQuantizationType
    for node in graph.nodes:
        node_type = node.node_type
        if node_type in WeightQuantizationType:
            node.weight_quantization_type = WeightQuantizationType[node_type]
            node.last_quantization_node = check_if_last_quantization_node(node.node_id, graph, WeightQuantizationType)
                
    return graph


def get_next_quantization_node(graph, node_id):
    """ 
    get next quantization node, return the first node found
    """
    node2outputs = graph.node2outputs
    input2nodes = graph.input2nodes
    nodes = graph.nodes

    if nodes[node_id].last_quantization_node:
        return -1

    outputs = node2outputs[node_id]
    for output in outputs:
        next_node_ids = input2nodes[output]
        for next_node_id in next_node_ids:
            if nodes[next_node_id].weight_quantization_type:
                return next_node_id
    
    for output in outputs:
        next_node_ids = input2nodes[output]
        for next_node_id in next_node_ids:
            return get_next_quantization_node(graph, next_node_id)


def get_quantization_siblings_of_add_child(graph, node_id):
    """ if child nodes contain 'ADD' node, return all quantization child nodes and its next_quantization_node"""
    node2outputs = graph.node2outputs
    input2nodes = graph.input2nodes
    nodes = graph.nodes
    node_output = node2outputs[node_id]
    quantization_siblings = []
    next_quantization_node = -1
    if len(node_output) > 1:
        raise NotImplementedError("only support nodes with one output!")
    child_nodes = input2nodes[node_output[0]]
    has_int7_add_child = False
    for child_node in child_nodes:
        if nodes[child_node].node_type == Operations['add'] and nodes[child_node].name in ADD_NODES_INT7:
            if has_int7_add_child:
                raise NotImplementedError("only support one int7 'ADD' child!")
            has_int7_add_child = True
            next_quantization_node = get_next_quantization_node(graph, child_node)
                
    if has_int7_add_child:
        for child_node in child_nodes:
            if nodes[child_node].weight_quantization_type:
                quantization_siblings.append(child_node)
    return has_int7_add_child, quantization_siblings, next_quantization_node


def quantize_weights(graph, activation_factors):
    """
    quantize weights to get quantized weights, bias, q_w_factors, deq_factors
    quantization_op_type: operations need to be quantized
    """ 
    assert WeightQuantizationGranularity in ['per_tensor', 'per_channel'], 'unsupported WeightQuantizationGranularity!'

    nodes = graph.nodes
    node2inputs = graph.node2inputs
    node2inputs = graph.node2inputs
    activations = graph.activations
    graph_inputs_num = len(graph.graph_inputs)

    for node in nodes:
        if node.weight_quantization_type:
            inputs = node2inputs[node.node_id]
            
            if len(inputs) == 1:
                # conv and fc only has one input
                node_input = inputs[0]
                if node_input < graph_inputs_num:
                    input_name = "input"
                else:
                    input_name = activations[node_input]
                
                if node.data.activation_factor:
                    q_a_factor = node.data.activation_factor
                else:
                    q_a_factor, _, __ = activation_factors[input_name]

                weight = node.data.weight
                bias = node.data.bias

                # quantization
                if WeightQuantizationGranularity == 'per_tensor' or node.node_type == Operations['matmul']:
                    upper = 1000
                    rw = np.max(np.abs(weight[np.abs(weight) < upper]))
                    q_w_factor = Q.get_quantizing_factor(rw, format=node.weight_quantization_type)
                    q_w_int8 = Q.quantizing_data(weight, q_w_factor, format=node.weight_quantization_type)

                    q_b_int32 = Q.quantizing_bias(bias, q_w_factor, q_a_factor)
                    deq_factor = 1 / (q_a_factor * q_w_factor)
                    node.data.deq_factor_with_s3 = 0.0
                else:  # per_channel
                    if node.node_type in [Operations['conv'], Operations['group_conv']]:
                        rw = []
                        for weight_elem in weight:
                            rw.append(compute_percentile(weight_elem.flatten(), 1.0))
                        q_w_factor = Q.get_quantizing_factor(rw, format=node.weight_quantization_type)
                        q_w_int8 = Q.quantizing_data(weight, q_w_factor, format=node.weight_quantization_type)
                        q_b_int32 = Q.quantizing_bias(bias, q_w_factor, q_a_factor)
                        deq_factor = [1 / (q_a_factor * qw) for qw in q_w_factor]
                        node.data.deq_factor_with_s3 = [0.0] * len(rw)

                node.data.weight_q = q_w_int8
                node.data.bias_q = q_b_int32
                node.data.activation_factor = q_a_factor
                node.data.weight_factor = q_w_factor
                node.data.deq_factor = deq_factor

                if not node.last_quantization_node:
                    quantization_siblings = []
                    if AddInputInt7:
                        has_int7_add_child, quantization_siblings, next_quantization_node = get_quantization_siblings_of_add_child(graph, node.node_id)
                        if not has_int7_add_child:
                            next_quantization_node = get_next_quantization_node(graph, node.node_id)
                    else:
                        next_quantization_node = get_next_quantization_node(graph, node.node_id)
                    
                    if next_quantization_node > 0:
                        output = activations[node2inputs[next_quantization_node][0]]
                        output_factor, _, __ = activation_factors[output]
                        
                        if AddInputInt7 and has_int7_add_child:
                            if ActivationQuantizationType == 'int8':
                                output_factor = output_factor / 2  # int7
                                if quantization_siblings:
                                    nodes[next_quantization_node].data.activation_factor = output_factor
                            else:
                                raise NotImplementedError("only support ActivationQuantizationType with int8!")
                        if WeightQuantizationGranularity == 'per_tensor':
                            deq_factor_with_s3 = deq_factor * output_factor
                        else:  # per_channel
                            deq_factor_with_s3 = [dq * output_factor for dq in deq_factor]
                        node.data.deq_factor_with_s3 = deq_factor_with_s3
                        for sibling in quantization_siblings:
                            # set activation_factor of quantization_siblings to that of next_quantization_node
                            nodes[sibling].data.activation_factor = output_factor
            else:
                raise NotImplementedError("quantization node should have one input node!")
    return graph


def check_if_last_quantization_node(node_id, graph, quantization_op_type):
    """check if there are other quantization nodes in the subgraph following this node
       if yes, output of this node is dequantize-requantized to int8 using deq_factor_with_s3
       otherwise, output of this node is dequantized to int32 using deq_factor
    """
    node2outputs = graph.node2outputs
    input2nodes = graph.input2nodes
    nodes = graph.nodes

    sub_outputs = node2outputs[node_id]
    for activation in sub_outputs:
        if activation in input2nodes:
            sub_node_ids = input2nodes[activation]
            for sub_node_id in sub_node_ids:
                # TODO: average_pool quantize or not?
                if nodes[sub_node_id].node_type == Operations['average_pool']:
                    return True
                elif nodes[sub_node_id].node_type in quantization_op_type:
                    return False
                else:
                    return check_if_last_quantization_node(sub_node_id, graph, quantization_op_type)
    return True

def select_best_quant_range(tensor, format):
    """
    select best quantization range for tensor, by minimizing mse between tensor X and quantized tensor X_quant
    """
    best_mse = 100000
    best_percentile = 0
    best_range = 0
    all_mse = []

    min_percentile = 999000
    step = 2
    max_percentile = 1000001
    percentiles = [x/10000 for x in range(min_percentile, max_percentile, step)]

    for percentile in (percentiles):
        tensor_range = np.percentile(np.abs(tensor).flatten(), percentile)
        tensor_q = Q.get_quantizing_factor(tensor_range, format)
        tensor_i8 = Q.quantizing_data(tensor, tensor_q, format)
        
        tensor_round = tensor_i8 / tensor_q
        mse_weight = 0.5 * np.sum((tensor - tensor_round) ** 2)
        # kldiv = torch.sum(torch.where(tensor != torch.tensor(0, dtype=torch.float), 
        # tensor * torch.log(tensor / (tensor_round+torch.tensor(1e-8,dtype=torch.float))), torch.tensor(0, dtype=torch.float)))
        # mse_weight = kldiv

        # diff_weight = torch.abs(tensor - tensor_round)
        all_mse.append(mse_weight)
        # print("percentile:{}, mse:{}, max diff:{}\n".format(percentile, mse_weight, torch.max(diff_weight)))

        if mse_weight < best_mse:
            best_percentile = percentile
            best_range = tensor_range
            best_mse = mse_weight
    print("best percentile:{}, best range:{} mse:{}\n\n".format(best_percentile, best_range, best_mse))
    return best_range




