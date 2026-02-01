# ResNet-50 Image Classification with ONNX Runtime

This console application demonstrates image classification using a pre-trained ResNet50 model.

**Workflow:**
1. Downloads and loads a pre-trained ResNet-50 ONNX model
2. Loads and preprocesses an image (resize to 224×224, normalize with ImageNet statistics)
3. Passes the image through ResNet convolutional layers and the network
4. Extracts 1000D logits vector from the final classification layer
5. Outputs raw logits for 1000 ImageNet classes

**Requirements:**
- Pre-trained model: [resnet50-v2-7.onnx](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx)
- Place the model file in the project root directory

**Model visualization:** Use [Netron](https://netron.app/) to explore the model architecture and layer names.