# 使用说明

## 1. 环境准备

### 操作系统

* **操作系统版本**：建议使用 Ubuntu 18.04 或更高版本。

### Python 版本

* **Python**：3.7 及以上版本

### 必需依赖包

请在开始前，确保已经安装以下依赖。你可以通过以下命令安装依赖：

```
bash


复制代码
pip install -r requirements.txt
```

以下是主要依赖项：

* `numpy >= 1.19.5`
* `pandas >= 1.1.5`
* `scikit-learn >= 0.24.1`
* `tensorflow >= 2.4.0` 或 `pytorch >= 1.7.0`（视选手使用的框架而定）
* `nvidia-container-toolkit`（确保 GPU 使用）

### 其他依赖

* **Docker**：需要安装 Docker 和 NVIDIA Container Toolkit 以支持 GPU 训练。

  安装 Docker 和 NVIDIA Container Toolkit：

  ```
  bash复制代码# 安装 Docker
  sudo apt update && sudo apt install docker.io
  
  # 安装 NVIDIA Container Toolkit
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

## 2. 运行说明

### 步骤 1：准备数据和代码

1. **`data/`目录**：请将训练数据放置于 `data` 目录下：
   * `data/raw_data/`：包含原始数据集文件，运行时将替换为官方数据。
   * `data/user_data/`：用于保存选手的模型文件、中间结果、数据、权重等。
   * `data/prediction_result/`：用于保存最终预测的结果，`result.csv`文件。
2. **`data/code/`目录**：请将程序的入口代码 `main.py` 放置于 `data/code/` 目录下。所有的训练和预测逻辑应在 `main.py` 中实现。

### 步骤 2：运行 `run.sh` 脚本

1. 请确保 `docker.tar` 镜像文件和 `run.sh` 脚本位于 `images` 目录下。

2. 进入 `images` 目录，运行以下命令启动 Docker 容器并开始训练：

   ```
   bash复制代码cd images
   ./run.sh
   ```

脚本会自动执行以下操作：

1. 检查 Docker 是否已安装并支持 GPU。
2. 检查是否已经存在名为 `test` 的 Docker 镜像，如果没有，则从 `docker.tar` 文件中加载镜像。
3. 挂载 `data` 目录，并启动 Docker 容器，在 `/data/code` 目录下执行 `main.py`。

### 步骤 3：查看输出结果

* **训练结果**：中间文件、模型权重等会被保存在 `data/user_data/` 目录下。
* **预测结果**：预测结果将保存为 `data/prediction_result/result.csv` 文件。

## 3. 注意事项

* 请确保在 `main.py` 中定义完整的训练、微调、生成模型和预测的流程。
* 如果运行时有任何特殊注意事项或依赖环境的问题，请在此文件中详细说明。
* 确保挂载的数据目录和代码目录符合要求。
* 如需使用多 GPU 训练，请在 `main.py` 中进行相关配置。

## 4. 特殊依赖或说明

* 如果你的代码依赖于特定的硬件配置（例如特定的 GPU 型号），请在此说明。
* 如有特殊的 Python 库版本要求，或需要额外的系统配置，请明确写出。

------

### 示例 `requirements.txt` 文件：

```
makefile复制代码numpy==1.19.5
pandas==1.1.5
scikit-learn==0.24.1
tensorflow==2.4.0  # 或者 PyTorch
torch==1.7.0  # 如果使用 PyTorch
```