import json

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# paths
model_path = "model_quant.onnx"
platform_path = "zc706.toml"

# create a parser
parser = Parser(backend="chisel", quant_mode="int", convert_gemm_to_conv=False)

# initialise network, and load a configuration
net = parser.onnx_to_fpgaconvnet(model_path, platform_path)
net.name = "zc706-mobilenet"

# remove squeeze layers
net.partitions[0].remove_squeeze()

# increase fine of depthwise layers
depthwise_fine = 3
net.partitions[0].graph.nodes["Conv_57"]["hw"].fine  = 9
net.partitions[0].graph.nodes["Conv_61"]["hw"].fine  = depthwise_fine
net.partitions[0].graph.nodes["Conv_69"]["hw"].fine  = depthwise_fine
net.partitions[0].graph.nodes["Conv_77"]["hw"].fine  = depthwise_fine
net.partitions[0].graph.nodes["Conv_85"]["hw"].fine  = depthwise_fine
net.partitions[0].graph.nodes["Conv_93"]["hw"].fine  = depthwise_fine
net.partitions[0].graph.nodes["Conv_101"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_109"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_117"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_125"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_133"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_141"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_149"]["hw"].fine = depthwise_fine
net.partitions[0].graph.nodes["Conv_157"]["hw"].fine = depthwise_fine

# increase coarse group of depthwise layers
depthwise_coarse_group = 8
net.partitions[0].graph.nodes["Conv_61"]["hw"].coarse_group  = depthwise_coarse_group//2
net.partitions[0].graph.nodes["Conv_69"]["hw"].coarse_group  = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_77"]["hw"].coarse_group  = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_85"]["hw"].coarse_group  = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_93"]["hw"].coarse_group  = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_101"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_109"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_117"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_125"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_133"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_141"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_149"]["hw"].coarse_group = depthwise_coarse_group
net.partitions[0].graph.nodes["Conv_157"]["hw"].coarse_group = depthwise_coarse_group

# increase pointwise coarse out
pointwise_coarse_out = 8
net.partitions[0].graph.nodes["Conv_57"]["hw"].coarse_out  = pointwise_coarse_out//2
net.partitions[0].graph.nodes["Conv_65"]["hw"].coarse_out  = pointwise_coarse_out
net.partitions[0].graph.nodes["Conv_73"]["hw"].coarse_out  = pointwise_coarse_out
net.partitions[0].graph.nodes["Conv_81"]["hw"].coarse_out  = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_89"]["hw"].coarse_out  = pointwise_coarse_out
net.partitions[0].graph.nodes["Conv_97"]["hw"].coarse_out  = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_105"]["hw"].coarse_out = pointwise_coarse_out
net.partitions[0].graph.nodes["Conv_113"]["hw"].coarse_out = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_121"]["hw"].coarse_out = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_129"]["hw"].coarse_out = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_137"]["hw"].coarse_out = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_145"]["hw"].coarse_out = pointwise_coarse_out*2
net.partitions[0].graph.nodes["Conv_153"]["hw"].coarse_out = pointwise_coarse_out
net.partitions[0].graph.nodes["Conv_161"]["hw"].coarse_out = pointwise_coarse_out*2

# update partitions
net.update_partitions()

# print per-layer latency estimates
for node in net.partitions[0].graph.nodes:
    print(f"{node}:\t {net.partitions[0].graph.nodes[node]['hw'].latency()}")

# print performance and resource estimates
print(f"predicted cycles: {net.get_cycle()}")
print(f"predicted latency (us): {net.get_latency()*1000000}")
print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")

# export out the configuration
net.save_all_partitions(f"{net.name}.json")

# add additional information to configuration
with open(f"{net.name}.json", "r") as f:
    config = json.load(f)

# buffer depths in and out
config["partition"][0]["buffer_depth_in"] = 16
config["partition"][0]["buffer_depth_out"] = 16

# input and output data type
config["partition"][0]["input_t"] = { "width": 16, "binary_point": 0 }
config["partition"][0]["output_t"] = { "width": 16, "binary_point": 0 }

# generate last signal counter width
config["partition"][0]["gen_last_width"] = 3

# save the updated config
with open(f"{net.name}.json", "w") as f:
    json.dump(config, f, indent=2)

