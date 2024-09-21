
# #模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat',cache_dir ='./model/')
model_dir = snapshot_download('NumberJys/glm4_9B_chat_review_1000_lora',cache_dir ='./lora/')

# model_dir = snapshot_download('qwen/qwen2-7b-instruct',cache_dir ='./model/')
# model_dir = snapshot_download('NumberJys/Qwen2_7B_instruct_1000_lora',cache_dir ='./lora/')
#
# model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2_5-20b-chat',cache_dir ='./model/')
# model_dir = snapshot_download('NumberJys/internlm2_5_20B_chat_1500_lora',cache_dir ='./lora/')
#
# model_dir = snapshot_download('llm-research/gemma-2-9b-it',cache_dir ='./model/')
# model_dir = snapshot_download('NumberJys/gemma-2-9b-it_review_1000_lora',cache_dir ='./lora/')


