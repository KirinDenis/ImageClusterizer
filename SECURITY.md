# Security Policy

## Overview

**Image Clusterizer** is a fully local, offline desktop application. It does not make network requests, does not transmit data externally, and stores all data (embeddings, thumbnails, settings) locally on your machine. There is no server component, no authentication, and no user accounts.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Older releases | No — please update to the latest version |

## Scope

Relevant security concerns for this project include:

- **Malicious ONNX model substitution** — the app loads `resnet50-v2-7.onnx` from the application directory. A tampered model could produce malicious output during inference.
- **Path traversal in folder scanning** — the app recursively scans user-selected folders. Symlinks or specially crafted folder structures should not cause unintended file access.
- **SQLite database integrity** — the local `vectors.db` stores embeddings and metadata. Tampering with it could cause unexpected behavior.
- **Malicious image files** — crafted image files (JPEG, PNG, BMP, WEBP) processed by System.Drawing could potentially trigger vulnerabilities in the image decoding layer.

## Reporting a Vulnerability

If you discover a security vulnerability in Image Clusterizer, please **do not open a public GitHub issue**.

Instead, report it privately:

1. Go to the [Security Advisories](https://github.com/KirinDenis/ImageClusterizer/security/advisories) tab on GitHub.
2. Click **"Report a vulnerability"** to open a private advisory.
3. Describe the vulnerability, steps to reproduce, and potential impact.

The maintainer will acknowledge the report within **7 days** and work towards a fix. Once a fix is released, the advisory will be published.

## Out of Scope

The following are **not** considered security vulnerabilities for this project:

- Issues requiring physical access to the user's machine
- Theoretical attacks with no practical exploit
- Issues in third-party dependencies (report those upstream: ONNX Runtime, MathNet.Numerics, SQLite)

Thank you for helping keep Image Clusterizer safe.
