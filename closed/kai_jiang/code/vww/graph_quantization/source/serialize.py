import numpy as np

from config import *
from utils import get_shift_n, save_list_to_text, save_twodimension_data
from graph import *

# below defines various position class, contains start position and offset pairs to access data
# e.g. pos = [start_pos1, offset1, start_pos2, offset2, start_pos3, offset3, start_pos4, offset4 ... ...]
class ConvPos:
    def __init__(self) -> None:
        self.weight_pos = []
        self.bias_pos = []
        self.weight_shape_pos = []
        self.bias_shape_pos = []
        self.quantization_factor_pos = []
        self.stride_pos = []
        self.pad_pos = []
        self.activation_func_pos = []
    
    def save_position(self, prefix):
        if (self.weight_pos):
            save_list_to_text(self.weight_pos, prefix + "data_conv_weight_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.bias_pos, prefix + "data_conv_bias_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.weight_shape_pos, prefix + "data_conv_weight_shape_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.bias_shape_pos, prefix + "data_conv_bias_shape_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.quantization_factor_pos, prefix + "data_conv_quantization_factor_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.stride_pos, prefix + "data_conv_stride_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.pad_pos, prefix + "data_conv_pad_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.activation_func_pos, prefix + "data_conv_activation_func_pos.txt", interval=100, ishex=False)

class MatmulPos:
    def __init__(self) -> None:
        self.weight_pos = []
        self.bias_pos = []
        self.weight_shape_pos = []
        self.bias_shape_pos = []
        self.quantization_factor_pos = []
        self.transa_pos = []
        self.transb_pos = []
        self.activation_func_pos = []
    
    def save_position(self, prefix):
        if (self.weight_pos):
            save_list_to_text(self.weight_pos, prefix + "data_matmul_weight_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.bias_pos, prefix + "data_matmul_bias_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.weight_shape_pos, prefix + "data_matmul_weight_shape_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.bias_shape_pos, prefix + "data_matmul_bias_shape_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.quantization_factor_pos, prefix + "data_matmul_quantization_factor_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.transa_pos, prefix + "data_matmul_transa_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.transb_pos, prefix + "data_matmul_transb_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.activation_func_pos, prefix + "data_matmul_activation_func_pos.txt", interval=100, ishex=False)

class TransposePos:
    def __init__(self) -> None:
        self.perm_pos = []
    
    def save_position(self, prefix):
        if (self.perm_pos):
            save_list_to_text(self.perm_pos, prefix + "data_transpose_perm_pos.txt", interval=100, ishex=False)

class ReshapePos:
    def __init__(self) -> None:
        self.shape_pos = []
    
    def save_position(self, prefix):
        if (self.shape_pos):
            save_list_to_text(self.shape_pos, prefix + "data_reshape_shape_pos.txt", interval=100, ishex=False)

class AveragepoolPos:
    def __init__(self) -> None:
        self.kernel_shape_pos = []
        self.strides_pos = []
    
    def save_position(self, prefix):
        if (self.kernel_shape_pos):
            save_list_to_text(self.kernel_shape_pos, prefix + "data_average_pool_kernel_shape_pos.txt", interval=100, ishex=False)
            save_list_to_text(self.strides_pos, prefix + "data_average_pool_strides_pos.txt", interval=100, ishex=False)

class SoftmaxPos:
    def __init__(self) -> None:
        self.axis_pos = []
    
    def save_position(self, prefix):
        if (self.axis_pos):
            save_list_to_text(self.axis_pos, prefix + "data_softmax_axis_pos.txt", interval=100, ishex=False)

class ConcatPos():
    def __init__(self) -> None:
        self.axis_pos = []
    
    def save_position(self, prefix):
        if (self.axis_pos):
            save_list_to_text(self.axis_pos, prefix + "data_concat_axis_pos.txt", interval=100, ishex=False)

class ModelData:
    def __init__(self, graph):
        self.graph = graph

        # config data
        if (ActivationQuantizationType == "int8"):
            self.activation_quantization_type = 0
        else:
            self.activation_quantization_type = 1
        
        if (WeightQuantizationGranularity == "per_tensor"):
            self.weight_quantization_granularity = 0
        else:
            self.weight_quantization_granularity = 1
        
        if (AddInputInt7):
            self.add_nodes_int7 = ADD_NODES_INT7
            self.add_nodes_int7_len = len(self.add_nodes_int7)
        else:
            self.add_nodes_int7 = []
            self.add_nodes_int7_len = 0

        # input data
        self.input_data = []

        # data array
        self.node_types = []
        self.weight_int8 = []
        self.weight_int32 = []
        self.config = []
        self.quantization_factor = []

        # data len
        self.weight_int8_len = self.weight_int32_len = self.quantization_factor_len = self.config_len = 0

        # position data
        self.conv_pos = ConvPos()
        self.matmul_pos = MatmulPos()
        self.transpose_pos = TransposePos()
        self.reshape_pos = ReshapePos()
        self.averagepool_pos = AveragepoolPos()
        self.softmax_pos = SoftmaxPos()
        self.concat_pos = ConcatPos()

        # graph data
        self.nodes_num = 0
        self.input_num = 0
        self.input2nodes_len = 0
        self.node2inputs_len = 0
        self.activations_num = 0
        self.node2outputs = []
        self.node2inputs = []
        self.input2nodes = []

        # load data
        self.load_operation_data()
        self.load_graph_information()

    def create_txt(self, prefix):
        # position data
        self.conv_pos.save_position(prefix)
        self.matmul_pos.save_position(prefix)
        self.transpose_pos.save_position(prefix)
        self.reshape_pos.save_position(prefix)
        self.concat_pos.save_position(prefix)
        self.averagepool_pos.save_position(prefix)
        self.softmax_pos.save_position(prefix)

        # data array
        save_list_to_text(self.node_types, prefix + "data_node_types.txt", interval=100, ishex=False)
        save_list_to_text(self.config, prefix + "data_config.txt", interval=100, ishex=False)
        save_list_to_text(self.weight_int8, prefix + "data_weight_int8.txt", interval=100, ishex=True)
        save_list_to_text(self.weight_int32, prefix + "data_weight_int32.txt", interval=100, ishex=True)
        save_list_to_text(self.quantization_factor, prefix + "data_quantization_factor.txt", interval=100, ishex=False)

        node2outputs_file = prefix + "data_node2outputs.txt"
        node2inputs_file = prefix + "data_node2inputs.txt"
        input2nodes_file = prefix + "data_input2nodes.txt"
        save_twodimension_data(node2outputs_file, self.node2outputs)
        save_twodimension_data(node2inputs_file, self.node2inputs)
        save_twodimension_data(input2nodes_file, self.input2nodes)
        
    def conv_serialize(self, node):
        data = node.data
        # note: check if depthwise conv2d, and transpose to (1, c, h, w) if true
        if node.node_type == Operations['group_conv'] == 1:
            weight_q = data.weight_q.transpose((1, 0, 2, 3))
        else:
            weight_q = data.weight_q
        self.weight_int8.extend(weight_q.flatten())
        self.conv_pos.weight_pos.append(self.weight_int8_len)
        self.conv_pos.weight_pos.append(np.prod(weight_q.shape))  # append offset of weight_q data
        self.weight_int8_len += np.prod(weight_q.shape)
        
        self.config.extend(weight_q.shape)
        self.conv_pos.weight_shape_pos.append(self.config_len)
        self.conv_pos.weight_shape_pos.append(len(weight_q.shape))
        self.config_len += len(weight_q.shape)

        self.weight_int32.extend(data.bias_q.flatten())
        self.conv_pos.bias_pos.append(self.weight_int32_len)
        self.conv_pos.bias_pos.append(np.prod(data.bias_q.shape))
        self.weight_int32_len += np.prod(data.bias_q.shape)
        
        self.config.extend(data.bias_q.shape)
        self.conv_pos.bias_shape_pos.append(self.config_len)
        self.conv_pos.bias_shape_pos.append(len(data.bias_q.shape))
        self.config_len += len(data.bias_q.shape)

        self.quantization_factor.extend(get_shift_n(data.activation_factor))
        if WeightQuantizationGranularity == 'per_tensor':
            self.quantization_factor.extend(get_shift_n(data.weight_factor))
            self.quantization_factor.extend(get_shift_n(data.deq_factor))
            self.quantization_factor.extend(get_shift_n(data.deq_factor_with_s3))
            self.conv_pos.quantization_factor_pos.append(self.quantization_factor_len)
            self.conv_pos.quantization_factor_pos.append(8)  # 4 factors, every factor contains two value: shifted int32 value and shift number
            self.quantization_factor_len += 8
        else:  # per_channel
            weight_factor_shift = [get_shift_n(wf) for wf in data.weight_factor]
            self.quantization_factor.extend(item for wf in weight_factor_shift for item in wf)
            deq_factor_shift = [get_shift_n(dq) for dq in data.deq_factor]
            self.quantization_factor.extend(item for dq in deq_factor_shift for item in dq)
            deq_factor_with_s3_shift = [get_shift_n(dqs3) for dqs3 in data.deq_factor_with_s3]
            self.quantization_factor.extend(item for dqs3 in deq_factor_with_s3_shift for item in dqs3)
            self.conv_pos.quantization_factor_pos.append(self.quantization_factor_len)
            """
            2 means elements number of activation_factor
            len(weight_factor_shift) * 2 * 3 means total elements number of weight_factor, deq_factor, deq_factor_with_s3
            """
            quantization_factor_item_len = 2 + len(weight_factor_shift) * 2 * 3
            self.conv_pos.quantization_factor_pos.append(quantization_factor_item_len)
            self.quantization_factor_len += quantization_factor_item_len
            
        # config
        self.config.extend(data.strides)
        self.conv_pos.stride_pos.append(self.config_len)
        self.conv_pos.stride_pos.append(len(data.strides))
        self.config_len += len(data.strides)

        self.config.extend(data.pads)
        self.conv_pos.pad_pos.append(self.config_len)
        self.conv_pos.pad_pos.append(len(data.pads))
        self.config_len += len(data.pads)

        self.config.append(node.activation_func_type)
        self.conv_pos.activation_func_pos.append(self.config_len)
        self.conv_pos.activation_func_pos.append(1)
        self.config_len += 1
    
    def matmul_serialize(self, node):
        data = node.data     
        self.weight_int8.extend(data.weight_q.flatten())
        self.matmul_pos.weight_pos.append(self.weight_int8_len)
        self.matmul_pos.weight_pos.append(np.prod(data.weight_q.shape))
        self.weight_int8_len += np.prod(data.weight_q.shape)
        
        self.config.extend(data.weight_q.shape)
        self.matmul_pos.weight_shape_pos.append(self.config_len)
        self.matmul_pos.weight_shape_pos.append(len(data.weight_q.shape))
        self.config_len += len(data.weight_q.shape)

        self.weight_int32.extend(data.bias_q.flatten())
        self.matmul_pos.bias_pos.append(self.weight_int32_len)
        self.matmul_pos.bias_pos.append(np.prod(data.bias_q.shape))
        self.weight_int32_len += np.prod(data.bias_q.shape)
        
        self.config.extend(data.bias_q.shape)
        self.matmul_pos.bias_shape_pos.append(self.config_len)
        self.matmul_pos.bias_shape_pos.append(len(data.bias_q.shape))
        self.config_len += len(data.bias_q.shape)

        self.quantization_factor.extend(get_shift_n(data.activation_factor))
        self.quantization_factor.extend(get_shift_n(data.weight_factor))
        self.quantization_factor.extend(get_shift_n(data.deq_factor))
        self.quantization_factor.extend(get_shift_n(data.deq_factor_with_s3))
        self.matmul_pos.quantization_factor_pos.append(self.quantization_factor_len)
        self.matmul_pos.quantization_factor_pos.append(8)
        self.quantization_factor_len += 8
            
        # config
        self.config.append(data.transa)
        self.matmul_pos.transa_pos.append(self.config_len)
        self.matmul_pos.transa_pos.append(1)
        self.config_len += 1

        self.config.append(data.transb)
        self.matmul_pos.transb_pos.append(self.config_len)
        self.matmul_pos.transb_pos.append(1)
        self.config_len += 1

        self.config.append(node.activation_func_type)
        self.matmul_pos.activation_func_pos.append(self.config_len)
        self.matmul_pos.activation_func_pos.append(1)
        self.config_len += 1
    
    def transpose_serialize(self, node):
        data = node.data
        self.config.extend(data.perm)
        self.transpose_pos.perm_pos.append(self.config_len)
        self.transpose_pos.perm_pos.append(len(data.perm))
        self.config_len += len(data.perm)
    
    def reshape_serialize(self, node):
        data = node.data
        self.config.extend(data.shape)
        self.reshape_pos.shape_pos.append(self.config_len)
        self.reshape_pos.shape_pos.append(len(data.shape))
        self.config_len += len(data.shape)
    
    def concat_serialize(self, node):
        data = node.data
        self.config.append(data.axis)
        self.concat_pos.axis_pos.append(self.config_len)
        self.concat_pos.axis_pos.append(1)
        self.config_len += 1
    
    def averagepool_serialize(self, node):
        data = node.data
        self.config.extend(data.kernel_shape)
        self.averagepool_pos.kernel_shape_pos.append(self.config_len)
        self.averagepool_pos.kernel_shape_pos.append(len(data.kernel_shape))
        self.config_len += len(data.kernel_shape)

        self.config.extend(data.strides)
        self.averagepool_pos.strides_pos.append(self.config_len)
        self.averagepool_pos.strides_pos.append(len(data.strides))
        self.config_len += len(data.strides)
    
    def softmax_serialize(self, node):
        data = node.data
        self.config.append(data.axis)
        self.softmax_pos.axis_pos.append(self.config_len)
        self.softmax_pos.axis_pos.append(1)
        self.config_len += 1
    
    def load_operation_data(self):
        """
        load operation weight, configs and quantization data
        """
        # assert WeightQuantizationGranularity in ['per_tensor', 'per_channel'], 'unsupported WeightQuantizationGranularity!'

        nodes = self.graph.nodes
        for node in nodes:
            self.node_types.append(node.node_type)
            if node.data:
                if node.node_type == Operations['conv'] or node.node_type == Operations['group_conv']:
                    self.conv_serialize(node)
                elif node.node_type == Operations['matmul']:
                    self.matmul_serialize(node)
                elif node.node_type == Operations['transpose']:
                    self.transpose_serialize(node)
                elif node.node_type == Operations['reshape']:
                    self.reshape_serialize(node)
                elif node.node_type == Operations['concat']:
                    self.concat_serialize(node)
                elif node.node_type == Operations['average_pool']:
                    self.averagepool_serialize(node)
                elif node.node_type == Operations['softmax']:
                    self.softmax_serialize(node)
                else:
                    raise NotImplementedError
    
    def load_graph_information(self):
        """load graph information
        """
        for _, inputs in self.graph.node2inputs.items():
            inputs_len = len(inputs)
            if inputs_len > self.node2inputs_len:
                self.node2inputs_len = inputs_len
        
        for _, nodes in self.graph.input2nodes.items():
            nodes_len = len(nodes)
            if nodes_len > self.input2nodes_len:
                self.input2nodes_len = nodes_len
        
        for _, outputs in self.graph.node2outputs.items():
            # outputs_len = len(outputs)
            outputs_in_c = [outputs[0]]  # there is only one output
            self.node2outputs.append(outputs_in_c)

        for _, inputs in self.graph.node2inputs.items():
            inputs_len = len(inputs)
            inputs_in_c = [inputs[i] if i < inputs_len else -1 for i in range(self.node2inputs_len)]  # convert to list with length node2inputs_len
            self.node2inputs.append(inputs_in_c)
        
        for _, nodes in self.graph.input2nodes.items():
            nodes_len = len(nodes)
            nodes_in_c = [nodes[i] if i < nodes_len else -1 for i in range(self.input2nodes_len)]
            self.input2nodes.append(nodes_in_c)
        
        self.nodes_num = self.graph.nodes_num
        self.input_num = len(self.input2nodes)
        self.activations_num = self.graph.activations_num
    
    def load_input(self, read_data, *args):
        self.input_data = read_data(*args)
        save_list_to_text(self.input_data, "text.txt")

    def create_source(self, prefix, save=False):
        cm = "#include \"model_data.h\"\n\n"
        cm += "const int IMAGE_DATA_DIMS[] = {"
        for sp in self.graph.graph_input_shape:
            cm += "{}, ".format(sp)
        cm = cm[:-2]
        cm += "};\n"
        
        if (self.add_nodes_int7_len > 0):
            cm += "const int ADD_NODES_INT7[ADD_NODES_INT7_LEN] = {"
            for elem in ADD_NODES_INT7:
                node_id = get_nodeid_by_name(elem)
                cm += "{}, ".format(node_id)
            cm = cm[:-2]
            cm += "};\n"
        
        cm += "const int NODE2OUTPUTS[NODE_NUM][1] = \n"
        cm += str(self.node2outputs).replace("[", "{").replace("]", "}")
        cm += ";\n"

        cm += "const int NODE2INPUTS[NODE_NUM][NODE2INPUTS_LEN] = \n"
        cm += str(self.node2inputs).replace("[", "{").replace("]", "}")
        cm += ";\n"

        cm += "const int INPUT2NODES[INPUT_NUM][INPUT2NODES_LEN] = \n"
        cm += str(self.input2nodes).replace("[", "{").replace("]", "}")
        cm += ";\n"

        cm += "const int NODE_TYPE[] = \n"
        cm += str(self.node_types).replace("[", "{").replace("]", "}")
        cm += ";\n"

        cm += "const int CONFIG[] = \n"
        cm += str(self.config).replace("[", "{").replace("]", "}")
        cm += ";\n"

        cm += "const int QUANTIZATION_FACTOR[] = \n"
        cm += str(self.quantization_factor).replace("[", "{").replace("]", "}")
        cm += ";\n"

        if self.conv_pos.weight_pos:
            cm += "const int CONV_WEIGHT_POS[] = \n"
            cm += str(self.conv_pos.weight_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
            
            cm += "const int CONV_BIAS_POS[] = \n"
            cm += str(self.conv_pos.bias_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int CONV_WEIGHT_SHAPE_POS[] = \n"
            cm += str(self.conv_pos.weight_shape_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int CONV_BIAS_SHAPE_POS[] = \n"
            cm += str(self.conv_pos.bias_shape_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int CONV_QUANTIZATION_FACTOR_POS[] = \n"
            cm += str(self.conv_pos.quantization_factor_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int CONV_STRIDE_POS[] = \n"
            cm += str(self.conv_pos.stride_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int CONV_PAD_POS[] = \n"
            cm += str(self.conv_pos.pad_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int CONV_ACTIVATION_FUNC_POS[] = \n"
            cm += str(self.conv_pos.activation_func_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
        
        if self.matmul_pos.weight_pos:
            cm += "const int MATMUL_WEIGHT_POS[] = \n"
            cm += str(self.matmul_pos.weight_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
            
            cm += "const int MATMUL_BIAS_POS[] = \n"
            cm += str(self.matmul_pos.bias_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int MATMUL_WEIGHT_SHAPE_POS[] = \n"
            cm += str(self.matmul_pos.weight_shape_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int MATMUL_BIAS_SHAPE_POS[] = \n"
            cm += str(self.matmul_pos.bias_shape_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int MATMUL_QUANTIZATION_FACTOR_POS[] = \n"
            cm += str(self.matmul_pos.quantization_factor_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int MATMUL_TRANSA_POS[] = \n"
            cm += str(self.matmul_pos.transa_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int MATMUL_TRANSB_POS[] = \n"
            cm += str(self.matmul_pos.transb_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int MATMUL_ACTIVATION_FUNC_POS[] = \n"
            cm += str(self.matmul_pos.activation_func_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

        if (self.transpose_pos.perm_pos):
            cm += "const int TRANSPOSE_PERM_POS[] = \n"
            cm += str(self.transpose_pos.perm_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
        
        if (self.reshape_pos.shape_pos):
            cm += "const int RESHAPE_SHAPE_POS[] = \n"
            cm += str(self.reshape_pos.shape_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
        
        if (self.softmax_pos.axis_pos):
            cm += "const int SOFTMAX_AXIS_POS[] = \n"
            cm += str(self.softmax_pos.axis_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
        
        if (self.averagepool_pos.kernel_shape_pos):
            cm += "const int AVERAGE_POOL_KERNEL_SHAPE_POS[] = \n"
            cm += str(self.averagepool_pos.kernel_shape_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int AVERAGE_POOL_STRIDES_POS[] = \n"
            cm += str(self.averagepool_pos.strides_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"

            cm += "const int AVERAGE_POOLING_DIVISOR[] = {1908874353, 34};  /* average pooling range is 3 * 3, so divisor is 1/9, and convert it to int32 */\n"
        
        if (self.concat_pos.axis_pos):
            cm += "const int CONCAT_AXIS_POS[] = \n"
            cm += str(self.concat_pos.axis_pos).replace("[", "{").replace("]", "}")
            cm += ";\n"
        
        interval = 100
        hex_input = [(str(hex(e)) + ",\n") if (idx % interval == 0 and idx > 0) else (str(hex(e)) + ", ") for idx, e in enumerate(self.input_data)]
        cm_input = ''.join(hex_input)
        cm_input = cm_input[:-2] + "\n"
        cm += "const unsigned char IMAGE_DATA[] = {\n" + cm_input + "};\n\n"

        hex_weight_int32 = [(str(hex(e)) + ",\n") if (idx % interval == 0 and idx > 0) else (str(hex(e)) + ", ") for idx, e in enumerate(self.weight_int32)]
        cm_weight_int32 = ''.join(hex_weight_int32)
        cm_weight_int32 = cm_weight_int32[:-2] + "\n"
        cm += "const int WEIGHT_INT32[] = {\n" + cm_weight_int32 + "};\n\n"

        hex_weight_int8 = [(str(hex(e)) + ",\n") if (idx % interval == 0 and idx > 0) else (str(hex(e)) + ", ") for idx, e in enumerate(self.weight_int8)]
        cm_weight_int8 = ''.join(hex_weight_int8)
        cm_weight_int8 = cm_weight_int8[:-2] + "\n"
        cm += "const char WEIGHT_INT8[] = {\n" + cm_weight_int8 + "};\n\n"

        if (save):
            with open(prefix + "model_data.c", mode = 'w', encoding='utf-8') as fh:
                fh.write(cm)

    def create_header(self, prefix, save=False):
        cm = "#pragma once\n\n"
        if (self.add_nodes_int7_len > 0):
            cm += "#define ADD_NODES_INT7_LEN {}\n".format(self.add_nodes_int7_len)
        cm += "#define WEIGHT_QUANTIZATION_GRANULARITY {}  // 0 means per_tensor, 1 means per_channel\n".format(self.weight_quantization_granularity)
        cm += "#define ACTIVATION_QUANTIZATION_TYPE {}  // 0 means int8, 1 means uint8\n".format(self.activation_quantization_type)
        cm += "#define NODE_NUM {}\n#define INPUT_NUM {}\n".format(self.nodes_num, self.input_num)
        cm += "#define INPUT2NODES_LEN {}\n#define NODE2INPUTS_LEN {}\n".format(self.input2nodes_len, self.node2inputs_len)
        cm += "#define ACTIVATIONS_NUM {}\n".format(self.activations_num)
        
        # define op
        if (self.conv_pos.weight_pos):
            cm += "#define CONV2D_OP\n#define DEPTHWISE_CONV2D_OP\n"
        if (self.matmul_pos.weight_pos):
            cm += "#define MATMUL_OP\n"
        if (self.transpose_pos.perm_pos):
            cm += "#define PERMUTE_OP\n"
        if (self.reshape_pos.shape_pos):
            cm += "#define VIEW_OP\n"
        if (self.softmax_pos.axis_pos):
            cm += "#define SOFTMAX_OP\n"
        if (self.averagepool_pos.kernel_shape_pos):
            cm += "#define AVERAGE_POOL_OP\n"
        if (self.concat_pos.axis_pos):
            cm += "#define CONCAT_OP\n"
        
        cm += "extern const unsigned char IMAGE_DATA[];\nextern const int IMAGE_DATA_DIMS[];\n"
        cm += "extern const int NODE_TYPE[];\nextern const char WEIGHT_INT8[];\nextern const int WEIGHT_INT32[];\n"
        cm += "extern const int CONFIG[];\nextern const int QUANTIZATION_FACTOR[];\n\n"
        if (self.add_nodes_int7_len > 0):
            cm += "extern const int ADD_NODES_INT7[];\n\n"
        
        cm += "extern const int NODE2OUTPUTS[NODE_NUM][1];\nextern const int NODE2INPUTS[NODE_NUM][NODE2INPUTS_LEN];\n"
        cm += "extern const int INPUT2NODES[INPUT_NUM][INPUT2NODES_LEN];\n"
        if (self.averagepool_pos.kernel_shape_pos):
            cm += "extern const int AVERAGE_POOLING_DIVISOR[]; /* average pooling divisor */\n"
        
        if (self.conv_pos.weight_pos):
            cm += "extern const int CONV_WEIGHT_POS[];\nextern const int CONV_BIAS_POS[];\nextern const int CONV_WEIGHT_SHAPE_POS[];\n"
            cm += "extern const int CONV_BIAS_SHAPE_POS[];\nextern const int CONV_QUANTIZATION_FACTOR_POS[];\n"
            cm += "extern const int CONV_STRIDE_POS[];\nextern const int CONV_PAD_POS[];\nextern const int CONV_ACTIVATION_FUNC_POS[];\n\n"
        if (self.matmul_pos.weight_pos):
            cm += "extern const int MATMUL_WEIGHT_POS[];\nextern const int MATMUL_BIAS_POS[];\nextern const int MATMUL_WEIGHT_SHAPE_POS[];\n"
            cm += "extern const int MATMUL_BIAS_SHAPE_POS[];\nextern const int MATMUL_QUANTIZATION_FACTOR_POS[];\n"
            cm += "extern const int MATMUL_TRANSA_POS[];\nextern const int MATMUL_TRANSB_POS[];\nextern const int MATMUL_ACTIVATION_FUNC_POS[];\n\n"
        if (self.transpose_pos.perm_pos):
            cm += "extern const int TRANSPOSE_PERM_POS[];\n\n"
        if (self.reshape_pos.shape_pos):
            cm += "extern const int RESHAPE_SHAPE_POS[];\n\n"
        if (self.softmax_pos.axis_pos):
            cm += "extern const int SOFTMAX_AXIS_POS[];\n\n"
        if (self.averagepool_pos.kernel_shape_pos):
            cm += "extern const int AVERAGE_POOL_KERNEL_SHAPE_POS[];\nextern const int AVERAGE_POOL_STRIDES_POS[];\n\n"
        if (self.concat_pos.axis_pos):
            cm += "extern const int CONCAT_POS_AXIS_POS[];\n\n"
        

        # cm += "extern const float PRIORS[];  /* SSD prior boxes */\n"
        # cm += "extern const int PRIORS_DIMS[];  /* dims of PRIORS */\n"
        # cm += "extern const float CENTER_VARIANCE;  /* center variance for ssd boxes */\n"
        # cm += "extern const float SIZE_VARIANCE;  /* size variance for ssd boxes */\n"
        # cm += "extern const float PROB_THRESHOLD;  /* threshold of probability fro selecting boxes */\n"
        # cm += "extern const float IOU_THRESHOLD;\nextern const int TOP_K;\nextern const int CANDIDATE;\n"       
        if (save):
            with open(prefix + "model_data.h", mode = 'w', encoding='utf-8') as fh:
                fh.write(cm)

    def create_cdata(self, prefix, save=False):
        self.create_source(prefix, save)
        self.create_header(prefix, save)
