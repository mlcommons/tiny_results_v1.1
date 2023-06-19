Operations = dict(
    conv = 0,
    group_conv = 1,
    relu = 2, 
    relu6 = 3,
    add = 4,
    transpose = 5,
    reshape = 6,
    concat = 7,
    average_pool = 8,
    softmax = 9,
    matmul = 10,
)

WeightQuantizationGranularity = 'per_tensor'  # per_tensor or per_channel

WeightQuantizationType = {
    Operations['conv']: 'int8',
    Operations['group_conv']: 'int8',
    Operations['matmul']: 'int8',
}

ActivationQuantizationType = 'uint8'  # int8(0 in c) or uint8(1 in c)

""" 
when AddInputInt7 is false, inputs' data type of 'ADD' is int32 when executing graph in c
when AddInputInt7 is true, inputs of nodes in ADD_NODES_INT7 need int7 quantization
"""
AddInputInt7 = False
ADD_NODES_INT7 = ['Add_5']

""" used to truncate graph"""
# END_NODE = "Concat_175"
# GRAPH_OUTPUT_NAME = ['716', '717']

# # mbv1
# END_NODE = "Concat_131"
# GRAPH_OUTPUT_NAME = ['424', '425']

# # mobileone
# END_NODE = "Concat_139"
# GRAPH_OUTPUT_NAME = ['604', '605']


END_NODE = "Identity"
GRAPH_OUTPUT_NAME = ['Identity']