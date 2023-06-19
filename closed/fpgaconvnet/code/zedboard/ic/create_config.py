import json

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# paths
model_path = "model_quant.onnx"
platform_path = "zedboard.toml"

# create a parser
parser = Parser(backend="chisel", quant_mode="int", convert_gemm_to_conv=False)

# initialise network, and load a configuration
net = parser.onnx_to_fpgaconvnet(model_path, platform_path)
net.name = "zedboard-resnet"

# change the fine factor of all layers
fine = 9
net.partitions[0].graph.nodes["Conv_21"]["hw"].fine = fine
net.partitions[0].graph.nodes["Conv_25"]["hw"].fine = fine
net.partitions[0].graph.nodes["Conv_29"]["hw"].fine = fine
net.partitions[0].graph.nodes["Conv_39"]["hw"].fine = fine
net.partitions[0].graph.nodes["Conv_43"]["hw"].fine = fine
net.partitions[0].graph.nodes["Conv_53"]["hw"].fine = fine
net.partitions[0].graph.nodes["Conv_57"]["hw"].fine = fine

# change the coarse out factor of all layers
coarse_out_base = 4
net.partitions[0].graph.nodes["Conv_21"]["hw"].coarse_out = 2
net.partitions[0].graph.nodes["Conv_25"]["hw"].coarse_out = 2*coarse_out_base
net.partitions[0].graph.nodes["Conv_29"]["hw"].coarse_out = 2*coarse_out_base
net.partitions[0].graph.nodes["Conv_36"]["hw"].coarse_out = coarse_out_base
net.partitions[0].graph.nodes["Conv_39"]["hw"].coarse_out = coarse_out_base
net.partitions[0].graph.nodes["Conv_43"]["hw"].coarse_out = 2*coarse_out_base
net.partitions[0].graph.nodes["Conv_50"]["hw"].coarse_out = coarse_out_base
net.partitions[0].graph.nodes["Conv_53"]["hw"].coarse_out = coarse_out_base
net.partitions[0].graph.nodes["Conv_57"]["hw"].coarse_out = 2*coarse_out_base

# update partitions
net.update_partitions()

for node in net.partitions[0].graph.nodes:
    print(f"{node}:\t {net.partitions[0].graph.nodes[node]['hw'].latency()}")

# print performance and resource estimates
print(f"predicted latency (us): {net.get_latency()*1000000}")
print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

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

# reverse order of eltwise layer streams in
for node in config["partition"][0]["layers"]:
    if node["name"] == "Add_32":
        node["streams_in"] = list(reversed(node["streams_in"]))

# turn off fusing relu for certain convolution layers
dont_fuse_relu_layers = [ "Conv_29", "Conv_36", "Conv_43", "Conv_50", "Conv_57" ]
for node in config["partition"][0]["layers"]:
    if node["name"] in dont_fuse_relu_layers:
        node["parameters"]["fuse_relu"] = False

# use DSPs for the following layers
dsp_layers = [ "Conv_21", "Conv_25", "Conv_43", "Conv_36", "Conv_50", "Conv_25" ]
for node in config["partition"][0]["layers"]:
    if node["type"] == "CONVOLUTION":
        if node["name"] in dsp_layers:
            node["parameters"]["use_dsp"] = True
        else:
            node["parameters"]["use_dsp"] = False

# save the updated config
with open(f"{net.name}.json", "w") as f:
    json.dump(config, f, indent=2)

