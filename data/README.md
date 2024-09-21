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

## 环境依赖

项目的运行依赖以下环境和工具,详细配置见 [requirements.txt](./requirements.txt)：

* Python 3.9.12
* CUDA:12.2、CUDNN:8902
* 主要库依赖：
  * `transformers==4.44.1`
  * `torch==2.3.1`
  * `chromadb==0.5.0`
  * `langchain`
  * `xtuner`
  * `lmdeploy==0.5.3`

可以使用以下命令安装所需的Python库：

```bash
pip install -r requirements.txt
```

## 代码说明

### `main.py`

主程序入口，负责项目的整体控制流程。包括数据加载、模型初始化、推理执行等。

### `infer.py`

推理脚本，用于执行模型的推理任务，生成最终的预测结果。

### `official_data_process.py`

官方数据处理脚本，负责将原始数据转化为适合模型输入的格式。

### `internlm2_5_chat_20b_qlora_alpaca_e3.py`

`XTuner` 框架微调配置，涉及超参数的设置。

### `run.sh`

自动化脚本，用于执行项目流程，包括数据处理、模型推理。

## 数据说明

### 数据结构

* **训练数据**：`round1_train_data.jsonl`
  包含用于模型训练的标注数据，格式为JSON Lines。
* **测试数据**：`round1_test_data.jsonl`
  包含用于模型测试的无标注数据，格式为JSON Lines。

### 外部数据

* **Chroma 向量数据库**：存储在 `data/external_data/chroma/` 目录下，用于加速数据检索和匹配,其数据来源于**训练集**，未使用外部数据，此处将其转为文本文件，进行向量化存储。

## 运行步骤

### 使用到的主要微调和量化工具

1. **微调工具** [**`XTuner`**](https://github.com/InternLM/xtuner) https://github.com/InternLM/xtuner
2. **量化工具** [**`LMDeploy`**](https://github.com/InternLM/lmdeploy) https://github.com/InternLM/lmdeploy

### 使用 `XTuner` 进行模型微调

#### 用于微调训练的数据格式

   通过脚本对官方训练集进行数据处理，得到满足 `XTuner` [微调格式的数据](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_format.md) ( `"system"`可省略 )。形如：

   ```json
    [
        {
            "conversation":[
                {
                    "system": "xxx",
                    "input": "xxx",
                    "output": "xxx"
                }
            ]
        },
       {
           "conversation": [
               {   "system": "xxx",
                   "input": "有一个英文到法文的词汇表，包含以下对应词汇:\n\n1. the -> le\n2. cat -> chat\n3. jumps -> sauts\n4. over -> sur\n5. moon -> lune\n6. cow -> vache\n7. plays -> jouer\n8. fiddle -> violon\n9. egg -> bougre\n10. falls -> des chutes\n11. off -> de\n12. wall -> mur\n\n根据这个词汇表，翻译以下英文句子成法文:选择题 1:\n英文句子 \"the cat jumps over the moon\" 翻译成法文是: A: le chat saute sur la lune; B: le chat sauts sur le lune; C: le sauts chat sur le lune; D: le chat sauts sur le lune",
                   "output": "D"
               }
           ]
       }
]
   ```

#### 微调训练命令

   ```bash
   # 训练
   xtuner train ../src/internlm2_5_chat_20b_qlora_alpaca_e3.py --work-dir ../models/temp/2_5_20b 
   
   # 转为hf格式
   export MKL_SERVICE_FORCE_INTEL=1
   export MKL_THREADING_LAYER=GNU
   xtuner convert pth_to_hf ../src/internlm2_5_chat_20b_qlora_alpaca_e3.py ../models/temp/2_5_20b/iter_220.pth ../models/temp/hf
   
   # 合并权重
   export MKL_SERVICE_FORCE_INTEL=1
   export MKL_THREADING_LAYER=GNU
   xtuner convert merge ../models/internlm2_5_chat_20b ../models/temp/hf ../models/internlm2_5_chat_20b_qlora --max-shard-size 2GB
   ```
   
### 使用 `LMdeploy` 进行量化

#### 量化命令

   ```bash
   # 8 bit 量化：
   export HF_MODEL=../models/internlm2_5_chat_20b_qlora
   export WORK_DIR=../models/internlm2_5_chat_20b_qlora_8bit
   lmdeploy lite smooth_quant $HF_MODEL --batch-size 10  --work-dir $WORK_DIR
   ```
   
   ```bash
   # 4 bit 量化：
   export HF_MODEL=/group_share/qlora_model/intern2_20b_qlora
   export WORK_DIR=/group_share/temp/intern2_20b_qlora_4bit
   lmdeploy lite auto_awq \
      $HF_MODEL \
     --calib-dataset 'ptb' \
     --calib-samples 128 \
     --calib-seqlen 2048 \
     --w-bits 4 \
     --w-group-size 128 \
     --batch-size 8 \
     --search-scale False \
     --work-dir $WORK_DIR
   ```

#### 量化效果

   量化后的模型大小及文件,整个推理大概需占用24G显存，主要包括量化模型以及向量化模型的加载（22G+1G+其他≈24G）

#### 量化后文件大小   
   ```bash
(test) root@root:../models/internlm2_5_chat_20b_qlora# du -sh ./
20G     ./

(test) root@root:../models/internlm2_5_chat_20b_qlora# tree -lh
.
├── [1.1K]  config.json
├── [8.6K]  configuration_internlm2.py
├── [ 123]  generation_config.json
├── [ 19M]  inputs_stats.pth
├── [ 83K]  modeling_internlm2.py
├── [ 31M]  outputs_stats.pth
├── [1.8G]  pytorch_model-00001-of-00011.bin
├── [1.9G]  pytorch_model-00002-of-00011.bin
├── [1.8G]  pytorch_model-00003-of-00011.bin
├── [1.8G]  pytorch_model-00004-of-00011.bin
├── [1.8G]  pytorch_model-00005-of-00011.bin
├── [1.8G]  pytorch_model-00006-of-00011.bin
├── [1.8G]  pytorch_model-00007-of-00011.bin
├── [1.8G]  pytorch_model-00008-of-00011.bin
├── [1.8G]  pytorch_model-00009-of-00011.bin
├── [1.8G]  pytorch_model-00010-of-00011.bin
├── [1.3G]  pytorch_model-00011-of-00011.bin
├── [ 45K]  pytorch_model.bin.index.json
├── [ 713]  special_tokens_map.json
├── [8.6K]  tokenization_internlm2.py
├── [1.4M]  tokenizer.model
└── [2.3K]  tokenizer_config.json
   ```

## 结果生成

  最终结果会保存在 `./submit/` 目录下，格式为 `JSON Lines` 文件。

### 运行模型推理

   通过 `run.sh` 脚本运行整个流程：

   ```bash
   (test) root@root:~# ./run.sh 
   已将测试集子问题写入：../data/external_data/infer_data.json
   已将训练集子问题写入：../data/external_data/train.json
   已将训练集子问题转为xtuner微调格式：../data/external_data/xtuner_train.json
   成功将 '../data/external_data/train.json' 转换为 '../data/external_data/train.txt'
   Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:06<00:00,  1.59it/s]
   模型加载成功！！
   100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 22192.08it/s]
   检测到数据库，加载中...
   数据库加载成功！！
   prompt processing:: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1328/1328 [00:37<00:00, 35.16it/s]
   所有prompt获取成功！！
   Submitting tasks: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1328/1328 [06:51<00:00,  3.23it/s]
   解答成功！！
   保存成功！！
   执行结束！
   总计耗时8.57 分钟
   ```

### 结果展示

推理后的结果文件将展示模型的预测效果，可以根据实际需求进行进一步的分析和优化。

