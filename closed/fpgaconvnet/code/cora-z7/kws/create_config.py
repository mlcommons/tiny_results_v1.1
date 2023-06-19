import json

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# paths
model_path = "model_quant.onnx"
platform_path = "cora-z7.toml"

# create a parser
parser = Parser(backend="chisel", quant_mode="int", convert_gemm_to_conv=False)

# initialise network, and load a configuration
net = parser.onnx_to_fpgaconvnet(model_path, platform_path)
net.name = "cora-z7-dscnn"

# remove squeeze layers and add split
net.partitions[0].remove_squeeze()

# update padding for first layer
net.partitions[0].graph.nodes["Conv_21"]["hw"]._pad = [5,1,5,2]

# change the fine factor of all layers
net.partitions[0].graph.nodes["Conv_21"]["hw"].fine = 10
net.partitions[0].graph.nodes["Conv_25"]["hw"].fine = 9
net.partitions[0].graph.nodes["Conv_33"]["hw"].fine = 9
net.partitions[0].graph.nodes["Conv_41"]["hw"].fine = 9
net.partitions[0].graph.nodes["Conv_49"]["hw"].fine = 9

net.update_partitions()

for node in net.partitions[0].graph.nodes:
    print(f"{node}:\t {net.partitions[0].graph.nodes[node]['hw'].latency()}")

# print performance and resource estimates
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
config["partition"][0]["gen_last_width"] = 6

# set point wise layers to not use dsps
non_dsp_layers = [ "Conv_21", "Conv_25" ]
for node in config["partition"][0]["layers"]:
    if node["name"] in non_dsp_layers:
        node["parameters"]["use_dsp"] = False

# save the updated config
with open(f"{net.name}.json", "w") as f:
    json.dump(config, f, indent=2)

