from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import PeftModel,LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import torch
import pandas as pd
from data_process import data_process
import os
def process_func_glm4(example):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    tokenizer = AutoTokenizer.from_pretrained('/data/user_data/model/glm-4-9b-chat', use_fast=True,trust_remote_code=True)
    instruction = tokenizer((f"[gMASK]<sop><|system|>\n您将担任评审的专家角色。<|user|>\n"
                            f"{example['instruction']+example['input']}<|assistant|>\n"
                            ).strip(), 
                            add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_func_qwen2(example):
    MAX_LENGTH = 2048    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    tokenizer = AutoTokenizer.from_pretrained('/data/user_data/model/qwen2-7b-instruct', use_fast=True,trust_remote_code=True)
    instruction = tokenizer(f"<|im_start|>system\n您将担任评审的专家角色<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_func_internlm2_5(example):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    tokenizer = AutoTokenizer.from_pretrained('/data/user_data/model/internlm2_5-20b-chat', use_fast=True,trust_remote_code=True)
    instruction = tokenizer(f"<s><|im_start|>system\n您将担任评审的专家角色<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False) 
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_func_gemma2(example):
    MAX_LENGTH = 2048    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    tokenizer = AutoTokenizer.from_pretrained('/data/user_data/model/gemma-2-9b-it', use_fast=True,trust_remote_code=True)
    instruction = tokenizer(f"<bos><start_of_turn>user\n{example['instruction'] + example['input']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens  
    response = tokenizer(f"{example['output']}<end_of_turn>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
def train_glm4(data_output_path,model_input_dir,model_output_dir):
    df = pd.read_json(data_output_path)
    ds = Dataset.from_pandas(df)
    model = AutoModelForCausalLM.from_pretrained(model_input_dir, device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir, use_fast=True,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_id = ds.map(process_func_glm4, remove_columns=ds.column_names)
    model.enable_input_require_grads()
    config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 现存问题只微调部分演示即可
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
    model = get_peft_model(model, config)
    args = TrainingArguments(
    output_dir= model_output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=50,
    num_train_epochs=5,
    save_steps=500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
    trainer.train()
    del model
    del tokenizer

def train_qwen2(data_output_path,model_input_dir,model_output_dir):
    df = pd.read_json(data_output_path)
    ds = Dataset.from_pandas(df)
    model = AutoModelForCausalLM.from_pretrained(model_input_dir, device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir, use_fast=True,trust_remote_code=True)
    tokenized_id = ds.map(process_func_qwen2, remove_columns=ds.column_names)
    model.enable_input_require_grads()
    config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
    model = get_peft_model(model, config)
    args = TrainingArguments(
    output_dir= model_output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=50,
    num_train_epochs=7,
    save_steps=500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
    trainer.train()
    del model
    del tokenizer


def train_internlm2_5(data_output_path,model_input_dir,model_output_dir):
    df = pd.read_json(data_output_path)
    ds = Dataset.from_pandas(df)
    model = AutoModelForCausalLM.from_pretrained(model_input_dir, device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir, use_fast=True,trust_remote_code=True)
    tokenized_id = ds.map(process_func_internlm2_5, remove_columns=ds.column_names)
    model.enable_input_require_grads()
    config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["wqkv", "wo", "w1", "w3","w2"],  # 现存问题只微调部分演示即可
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
    model = get_peft_model(model, config)
    args = TrainingArguments(
    output_dir= model_output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=50,
    num_train_epochs=7,
    save_steps=500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
    trainer.train()
    del model
    del tokenizer

def train_gemma2(data_output_path,model_input_dir,model_output_dir):
    df = pd.read_json(data_output_path)
    ds = Dataset.from_pandas(df)
    model = AutoModelForCausalLM.from_pretrained(model_input_dir, device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_input_dir, use_fast=True,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    tokenized_id = ds.map(process_func_gemma2, remove_columns=ds.column_names)
    model.enable_input_require_grads()
    config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
    model = get_peft_model(model, config)
    args = TrainingArguments(
    output_dir= model_output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=50,
    num_train_epochs=7,
    save_steps=500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
    trainer.train()
    del model
    del tokenizer

# if __name__ == "__main__":
#     os.system('apt install git')
#     os.system('apt install git-lfs')
    
#     os.system('git clone https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git /data/user_data/model/glm-4-9b-chat')
   
#     os.system('git clone https://www.modelscope.cn/qwen/qwen2-7b-instruct.git /data/user_data/model/qwen2-7b-instruct')
   
#     os.system('git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2_5-20b-chat.git /data/user_data/model/internlm2_5-20b-chat')

#     os.system('git clone https://www.modelscope.cn/llm-research/gemma-2-9b-it.git /data/user_data/model/gemma-2-9b-it')
   
    input_path = '/data/raw_data/train.csv'
    data_output_path = '/data/user_data/train_data.csv'
    
    data_process(input_path,data_output_path)
    
    glm4_model_path = '/data/user_data/model/glm-4-9b-chat'
    glm4_lora_path = '/data/user_data/sft_lora/glm4'
    # glm4_lora_path = '/group_share/temp'
    train_glm4(data_output_path,model_input_dir=glm4_model_path,model_output_dir=glm4_lora_path)
    torch.cuda.empty_cache()
    
    qwen2_model_path = '/data/user_data/model/qwen2-7b-instruct'
    qwen2_lora_path = '/data/user_data/sft_lora/qwen2'
    # qwen2_lora_path = '/group_share/temp1'
    train_qwen2(data_output_path,model_input_dir=qwen2_model_path,model_output_dir=qwen2_lora_path)
    torch.cuda.empty_cache()
    
    internlm2_5_model_path = '/data/user_data/model/internlm2_5-20b-chat'
    internlm2_5_lora_path = '/data/user_data/sft_lora/internlm2_5'
    # internlm2_5_lora_path = '/group_share/temp2'
    train_internlm2_5(data_output_path,model_input_dir=internlm2_5_model_path,model_output_dir=internlm2_5_lora_path)
    torch.cuda.empty_cache()
    
    gemma2_model_path = '/data/user_data/model/gemma-2-9b-it'
    gemma2_lora_path = '/data/user_data/sft_lora/gemma2'
    # gemma2_lora_path = '/group_share/temp4'
    train_gemma2(data_output_path,model_input_dir=gemma2_model_path,model_output_dir=gemma2_lora_path)
    torch.cuda.empty_cache()
    
    
    
