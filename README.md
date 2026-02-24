# Image Clusterizer

**🚧 Status:** Early Stage Development / Proof of Concept

This project is under active development and used for:
- Validating technology choices
- Testing architectural approaches
- Experimenting with new technologies

⚠️ Not recommended for production use.


## Step by Step 

A practical walkthrough of ResNet50 inference, feature extraction, similarity search, and dimensionality reduction using PCA.

### Step 1 — ResNet50 Logits (1000D vectors)

**`Polygon\1. ResNet50_GetLogits_Test`**

Raw output of the ResNet50 network — a 1000-dimensional vector of logits (one value per ImageNet class). These are unnormalized scores before softmax. The class with the highest logit is the predicted label.

### Step 2 — Feature Extraction (Embedding vectors)

**`Polygon\2. ResNet50_GetEmbedding_Test`**

Instead of taking the final classification layer output, we extract the penultimate layer — the **embedding vector** (also called feature vector). This is a dense representation of the image content, not tied to any specific class.

### Step 3 — Cosine Similarity Search

**`Polygon\3. ResNet50_Image_similarity_search_test`**

Compares images by measuring the angle between their vectors in high-dimensional space. Works for both logits (1000D) and embeddings (2048D).

### Step 4 — 2D PCA Visualization

**`Polygon\4. ResNet50_2D_PCA_test`**

Reduces high-dimensional vectors (1000D logits or 2048D embeddings) down to 2D for visualization, using Principal Component Analysis via SVD.

### Step 5. Image Similarity with Sparse Embeddings

**`Polygon\5. ResNet50_Sparse_Dot_Product_test`**

Demonstration of sparse vector representation for image similarity search using ResNet50 embeddings. Shows how to reduce 2048-dimensional vectors to sparse format (top-N values) and compute cosine similarity efficiently.

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

