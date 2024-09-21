from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

def load_model(model_path,lora_path):
# model_path = '/group_share/glm-4-9b-chat/ZhipuAI/glm-4-9b-chat'
# lora_path = '/group_share/glm4_9B_chat_lora/checkpoint-1000'
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,trust_remote_code=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
    
    # # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    return model,tokenizer