#!/bin/bash

# 确保脚本在 images 目录下执行
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || { echo "无法切换到 images 目录"; exit 1; }

# 检查 Docker 是否安装
if ! docker --version &> /dev/null
then
    echo "Docker 未安装，请先安装 Docker。"
    exit 1
fi

# 检查 Docker 是否支持 GPU
if ! docker run --gpus all --help > /dev/null 2>&1; then
    echo "你的 Docker 安装不支持 GPU。请安装 NVIDIA Container Toolkit。"
    exit 1
fi

# 变量定义
DOCKER_IMAGE="test"  # 默认的 Docker 镜像名称
TAR_FILE="docker.tar"  # 镜像的 tar 文件路径 (在 images 目录下)
DATA_DIR="$(pwd)/../data"  # data 目录位于上一级目录

# 检查镜像是否已存在
if [[ "$(docker images -q $DOCKER_IMAGE 2> /dev/null)" == "" ]]; then
    echo "Docker 镜像未找到，正在从 tar 文件导入..."

    # 检查 tar 文件是否存在
    if [ -f "$TAR_FILE" ]; then
        # 导入 Docker 镜像
        docker load -i $TAR_FILE
        
        # 获取刚刚导入的镜像 ID
        IMPORTED_IMAGE_ID=$(docker images -q | head -n 1)

        # 将导入的镜像重命名为 'test:latest'
        docker tag $IMPORTED_IMAGE_ID $DOCKER_IMAGE:latest
        
        echo "镜像已导入并重命名为 'test:latest'。"
    else
        echo "未找到 tar 文件：$TAR_FILE，请确保 tar 文件存在于 images 目录下。"
        exit 1
    fi
else
    echo "Docker 镜像 'test' 已存在，跳过导入步骤。"
fi

# 运行 Docker 容器，挂载 data 目录并使用所有 GPU
docker run --gpus all \
    --pull=never \
    --mount type=bind,source=$DATA_DIR,target=/data \
    $DOCKER_IMAGE \
    bash -c "cd /data/code && python sft.py"
