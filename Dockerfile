# 使用 nvidia/cuda:11.8.0-devel-ubuntu20.04 作为基础镜像
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装必要的工具和库（不更新软件源）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libopencv-dev \
    libeigen3-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libepoxy-dev \
    libboost-all-dev \
    pkg-config \
    software-properties-common \
    curl \
    gedit \
    libssl-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖和 SuperPoint、LightGlue 的依赖
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    torch==1.9.1 \
    torchvision==0.10.1 \
    numpy==1.18.1 \
    opencv-python==4.1.2.30 \
    matplotlib==3.1.3 \
    kornia==0.6.11

# 手动安装 Pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd ../.. && rm -rf Pangolin

# 设置新的工作目录
WORKDIR /sly_slam

# 将新的项目源代码复制到容器中
COPY . /sly_slam

# 安装 libtorch (GPU 版本)
#RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu118.zip && \
#    unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cu118.zip && \
#    mv libtorch /sly_slam/Thirdparty/ && \
#    rm libtorch-cxx11-abi-shared-with-deps-2.4.1+cu118.zip

# 安装 ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && \
    apt-get install -y ros-noetic-desktop-full && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    rm -rf /var/lib/apt/lists/*

# 配置 ROS 环境
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/sly_slam/Examples/ROS" >> ~/.bashrc

CMD ["bash"]
