# Image Clusterizer

**🚧 Status:** Early Stage Development / Proof of Concept

This project is under active development and used for:
- Validating technology choices
- Testing architectural approaches
- Experimenting with new technologies

⚠️ Not recommended for production use.

## Description
...

A Windows desktop application for organizing and clustering images using artificial intelligence.

## Description

Image Clusterizer automatically organizes large collections of images by grouping visually similar photos together. The application analyzes image content using deep learning and creates intelligent clusters based on visual features like objects, scenes, colors, and composition.

## What It Does

- Analyzes your image collections using AI
- Groups similar images into clusters automatically
- Helps organize photos without manual tagging
- Identifies visual patterns and similarities across your image library
- Provides fast batch processing for large photo collections

## Technology

**Platform:** Windows desktop application

**Framework:** .NET 8.0 with WinUI 3

**Architecture:** MVVM pattern

**AI Model:** ResNet-50 (Convolutional Neural Network)

The application uses the pre-trained ResNet-50 model from the ONNX Model Zoo. ResNet-50 is a 50-layer deep convolutional neural network trained on ImageNet dataset containing over 14 million images across 1000 categories. The model extracts high-level visual features from images which are then used to compute similarity and group related photos together.

Model source: https://github.com/onnx/models/tree/main

ResNet-50 provides robust image understanding capabilities including:
- Object recognition
- Scene classification
- Visual feature extraction
- Content-based image similarity

## How It Works

1. Load your image folder
2. ResNet-50 processes each image and generates feature embeddings
3. Clustering algorithms group images with similar features
4. Review and manage the automatically created clusters

## License

MIT

