## Hi there 👋


# DANI-SLAM

DANI-SLAM is a **dynamic-aware SLAM system** that integrates deep feature extraction, robust feature matching, and neural implicit mapping to achieve accurate localization and high-quality mapping in dynamic environments.

The system is designed to handle challenging scenarios with moving objects by combining learning-based perception modules with a classical SLAM framework.

---

## Overview

DANI-SLAM follows a modular pipeline consisting of:

- **Dynamic object awareness** for suppressing features from non-static regions
- **Learning-based feature extraction and matching**
- **Robust camera tracking and optimization**
- **Neural implicit scene representation for mapping**

The overall framework aims to improve robustness and map quality in complex real-world dynamic scenes.

---

## Code Availability

This repository contains the **complete implementation** of our proposed SLAM system.

To comply with the double-blind review policy and the ongoing paper submission process,  
some components (e.g., pretrained models, large vocabulary files, and detailed configuration files) are **temporarily withheld**.

**The full source code, pretrained models, vocabulary files, and all related resources will be publicly released upon paper acceptance.**

---

## Installation

### Prerequisites

- Ubuntu 20.04
- C++17
- CMake ≥ 3.10
- OpenCV
- CUDA (optional, for learning-based modules)
- PyTorch / LibTorch (if required)

### Build

```bash
git clone https://github.com/DANI-SLAM/DANI-SLAM.git
cd DANI-SLAM
chmod +x build.sh
./build.sh

