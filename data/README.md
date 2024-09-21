# 基于大模型的文本内容智能评判

比赛官网：(https://www.datafountain.cn/competitions/1032)

## 项目结构

```bash
|-- image
        |-- 打包后的docker镜像
        |-- run.sh
        |-- README.md
|-- data
        |-- raw_data
                |-- train.csv          # A榜训练集
                |-- test-B.xlsx        # B榜测试集
        |-- user_data
                |-- model  # 官方原始模型文件
                         |-- glm-4-9b-chat    
                         |-- qwen2-7b-instruct
                         |-- internlm2_5-20b-chat
                         |-- gemma-2-9b-it
                |-- lora  # 模型微调的lora文件
                         |-- glm4_9B_chat_review_1000_lora  # checkpoint:1000steps
                         |-- Qwen2_7B_instruct_1000_lora    # checkpoint:1000steps
                         |-- internlm2_5_20B_chat_1500_lora  # checkpoint:1500steps
                         |-- gemma-2-9b-it_review_1000_lora  # checkpoint:1000steps
                |-- glm4_9b_1000.csv  # 使用微调好的glm4_9b模型推理生成文件
                |-- qwen2_7b_1000.csv  # 使用微调好的qwen2_7b模型推理生成文件
                |-- internlm2.5_20b_1500.csv   # 使用微调好的internlm2.5_20b模型推理生成文件
                |-- gemma2_9b_1000.csv   # 使用微调好的gemma2_9b模型推理生成文件
        |-- prediction_result
                |-- result.csv  # 四个模型投票，针对B榜测试集的预测结果
        |-- code
                |-- data_process.py  # 处理成适合模型训练的数据格式脚本
                |-- sft.py  # 模型微调脚本
                |-- prompt.py  # 推理时用的prompt脚本
                |-- model.py  # model、tokenizer加载的脚本
                |-- infer.py  # 模型推理函数的脚本
                |-- main.py  # 模型推理的主函数
        |-- README.md

```

## **四路投票**模型推理

使用以下命令创建虚拟环境，并安装所需的Python库，最后运行main.py,进行推理复现

```bash
conda create -n llm python=3.10
conda activate llm
pip install pandas transformers peft accelerate sentencepiece datasets tiktoken openpyxl protobuf einops
python /data/code/main.py
```

## 代码说明

### `main.py`

主程序入口，负责项目的整体控制流程。包括数据加载、模型初始化、推理执行等。

### `infer.py`

推理脚本，用于执行模型的推理任务，生成最终的预测结果。

### `data_process.py`

数据处理脚本，负责将原始数据转化为适合模型输入的格式。

### `prompt.py`

提示词构造脚本

### `sft.py`

模型有监督微调脚本，用于模型的lora微调。

## 数据说明


- 训练数据：`train.csv`
  A榜训练数据，用于模型训练
- 测试数据：`test-B.xlsx`
  B榜测试数据，用于模型推理

## 模型说明





### 结果输出

推理后的结果文件将输出到 **/data/prediction_result/result.csv**

