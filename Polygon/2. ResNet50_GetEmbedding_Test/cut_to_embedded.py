# pip install onnx
import onnx

model = onnx.load("resnet50-v2-7.onnx")


embedding_node = None
for node in model.graph.node:
    if "pool1" in node.name or "avgpool" in node.name:
        embedding_node = node
        break

if embedding_node:
    # Delete all other layers agter embedding 
    model.graph.output[0].name = embedding_node.output[0]
    
onnx.save(model, "resnet50-embedding-only.onnx")