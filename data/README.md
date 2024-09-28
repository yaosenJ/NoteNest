# 基于大模型的文本内容智能评判

比赛官网：(https://www.datafountain.cn/competitions/1032)

## 项目结构

```bash
|-- image
        |-- docker.tar  #打包后的docker镜像
        |-- run_sft.sh  #一键训练脚本
        |-- run_infer.sh  #一键推理脚本
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
                |-- lora  # 模型已微调好的lora文件
                         |-- glm4_9B_chat_review_1000_lora  # checkpoint:1000steps
                         |-- Qwen2_7B_instruct_1000_lora    # checkpoint:1000steps
                         |-- internlm2_5_20B_chat_1500_lora  # checkpoint:1500steps
                         |-- gemma-2-9b-it_review_1000_lora  # checkpoint:1000steps
                |-- sft_lora  # 模型微调过程中产生的lora文件
                         |-- glm4
                                |-- checkpoint-1000  #最优步数
                         |-- qwen2
                                |-- checkpoint-1000  #最优步数
                         |-- internlm2_5
                                |-- checkpoint-1500  #最优步数
                         |-- gemma2
                                |-- checkpoint-1000  #最优步数
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
推理策略：
- 依次使用gemma2_9b、qwen2_7b、internlm2.5_20b、glm4_9b模型进行推理，将结果保存到对应文件中，最后使用四路投票策略，将四个模型的结果进行投票，得到最终结果。
- 每个模型推理结束，使用`torch.cuda.empty_cache()`,`del model`,`del tokenizer`释放显存，继续下个模型推理。

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

当运行sft脚本，会依次下载如下模型。并分别下载到 **/data/user_data/model/glm-4-9b-chat**、**/data/user_data/model/qwen2-7b-instruct**、**/data/user_data/model/internlm2_5-20b-chat**、**/data/user_data/model/gemma-2-9b-it**
- glm-4-9b-chat    [https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat](https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat)
- qwen2-7b-instruct   [https://www.modelscope.cn/qwen/qwen2-7b-instruct](https://www.modelscope.cn/qwen/qwen2-7b-instruct)
- internlm2_5-20b-chat  [https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2_5-20b-chat](https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2_5-20b-chat)
- gemma-2-9b-it  [https://www.modelscope.cn/llm-research/gemma-2-9b-it](https://www.modelscope.cn/llm-research/gemma-2-9b-it)

训练策略：
- 依次使用gemma2_9b、qwen2_7b、internlm2.5_20b、glm4_9b模型进行训练。
- 每个模型训练结束，使用`torch.cuda.empty_cache()`,`del model`,`del tokenizer`释放显存，继续下个模型训练。
- 每个模型训练结束，将模型checkpoint保存到对应模型目录下。
  
在微调后，分别选择glm-4-9b-chat的checkpoint-1000,qwen2-7b-instruct的checkpoint-1000，internlm2_5-20b-chat的checkpoint-1500，gemma-2-9b-it的checkpoint-1000
最后上传到
- [https://www.modelscope.cn/models/NumberJys/glm4_9B_chat_review_1000_lora](https://www.modelscope.cn/models/NumberJys/glm4_9B_chat_review_1000_lora)
- [https://www.modelscope.cn/models/NumberJys/Qwen2_7B_instruct_1000_lora](https://www.modelscope.cn/models/NumberJys/Qwen2_7B_instruct_1000_lora)
- [https://www.modelscope.cn/models/NumberJys/internlm2_5_20B_chat_1500_lora](https://www.modelscope.cn/models/NumberJys/internlm2_5_20B_chat_1500_lora)
- [https://www.modelscope.cn/models/NumberJys/gemma-2-9b-it_review_1000_lora](https://www.modelscope.cn/models/NumberJys/gemma-2-9b-it_review_1000_lora)

若推理时，注销官网模型下载的代码，因为在微调的时候已经下载，所以只需要下载上面上传的lora文件即可。注意：若直接推理，不在重新微调，就不需要注释。

```python
# os.system('git clone https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git /data/user_data/model/glm-4-9b-chat')
os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/glm4_9B_chat_review_1000_lora.git /data/user_data/lora/glm4_9B_chat_review_1000_lora')

# os.system('git clone https://www.modelscope.cn/qwen/qwen2-7b-instruct.git /data/user_data/model/qwen2-7b-instruct')
os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/Qwen2_7B_instruct_1000_lora.git /data/user_data/lora/Qwen2_7B_instruct_1000_lora')

# os.system('git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2_5-20b-chat.git /data/user_data/model/internlm2_5-20b-chat')
os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/internlm2_5_20B_chat_1500_lora.git /data/user_data/lora/internlm2_5_20B_chat_1500_lora')

# os.system('git clone https://www.modelscope.cn/llm-research/gemma-2-9b-it.git /data/user_data/model/gemma-2-9b-it')
os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/gemma-2-9b-it_review_1000_lora.git /data/user_data/lora/gemma-2-9b-it_review_1000_lora')
```

### 结果输出

投票机制：

- 多数票：如果某个分数获得了 3 票或 4 票（即 most_common[0][1] >= 3），将该分数作为最终预测结果。
- 平局：如果有两个分数各获得了 2 票（即 most_common[0][1] == 2 且 most_common[1][1] == 2），则使用 glm4 的预测分数作为默认值。
- 没有平局：如果只有一个分数获得了 2 票，将其作为最终预测结果。
- 无多数票：如果没有明显的多数票，则默认使用 glm4 的预测分数作为最终结果。

四路投票推理后的结果文件将输出到 **/data/prediction_result/result.csv**



