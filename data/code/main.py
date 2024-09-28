from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from collections import Counter
from infer import predict_glm4,predict_qwen2,predict_internlm2_5,predict_gemma2
from prompt import get_prompt1,get_prompt2,get_prompt3
from model import load_model
import torch
import os

# 模型下载
# os.system('apt install git')
# os.system('apt install git-lfs')

# os.system('git clone https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git /data/user_data/model/glm-4-9b-chat')
# os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/glm4_9B_chat_review_1000_lora.git /data/user_data/lora/glm4_9B_chat_review_1000_lora')

# os.system('git clone https://www.modelscope.cn/qwen/qwen2-7b-instruct.git /data/user_data/model/qwen2-7b-instruct')
# os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/Qwen2_7B_instruct_1000_lora.git /data/user_data/lora/Qwen2_7B_instruct_1000_lora')

# os.system('git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2_5-20b-chat.git /data/user_data/model/internlm2_5-20b-chat')
# os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/internlm2_5_20B_chat_1500_lora.git /data/user_data/lora/internlm2_5_20B_chat_1500_lora')

# os.system('git clone https://www.modelscope.cn/llm-research/gemma-2-9b-it.git /data/user_data/model/gemma-2-9b-it')
# os.system('git clone https://oauth2:LAAN_szcahZCFtryFxBs@www.modelscope.cn/NumberJys/gemma-2-9b-it_review_1000_lora.git /data/user_data/lora/gemma-2-9b-it_review_1000_lora')


"""
环境
pip install pandas transformers peft accelerate sentencepiece datasets tiktoken openpyxl protobuf einops
"""
def main1():   
    glm4_model_path = '/data/user_data/model/glm-4-9b-chat'
    glm4_lora_path = '/data/user_data/lora/glm4_9B_chat_review_1000_lora'
    glm4_model,glm4_tokenizer = load_model(glm4_model_path,glm4_lora_path)
    print("模型加载成功！")
    label = []
    for i in tqdm(range(len(test_df))):
        test_item = test_df.loc[i]
        if test_item['评判维度'] == "选择题":
            test_input = f"{test_item['创作要求']}。{test_item['待评判内容']}"
            # print(test_input)
            prompt= get_prompt1(test_input)
            label.append(int(predict_glm4(prompt,glm4_model,glm4_tokenizer)))
        elif test_item['评判维度'] == "流畅性":
            test_input = f"模型创作:{test_item['待评判内容']}"
            # print(test_input)
            prompt= get_prompt2(test_input)
            label.append(int(predict_glm4(prompt,glm4_model,glm4_tokenizer)))
        else:
            test_input = f"创作要求:{test_item['创作要求']}。模型创作:{test_item['待评判内容']}"
            prompt= get_prompt3(test_input)
            # print(test_input)
            label.append(int(predict_glm4(prompt,glm4_model,glm4_tokenizer)))
        # exit()
    test_df['预测分数'] = label
    submit = test_df[['数据编号', '评判维度', '预测分数']]
    os.makedirs(os.path.dirname(path1), exist_ok=True)
    submit.to_csv(path1, index=False)
    del model
    del tokenizer


def main2():   
    qwen2_model_path = '/data/user_data/model/qwen2-7b-instruct'
    qwen2_lora_path = '/data/user_data/lora/Qwen2_7B_instruct_1000_lora'
    qwen2_model,qwen2_tokenizer = load_model(qwen2_model_path,qwen2_lora_path)
    label = []
    for i in tqdm(range(len(test_df))):
        test_item = test_df.loc[i]
        if test_item['评判维度'] == "选择题":
            test_input = f"{test_item['创作要求']}。{test_item['待评判内容']}"
            prompt= get_prompt1(test_input)
            label.append(int(predict_qwen2(prompt,qwen2_model,qwen2_tokenizer)))
        elif test_item['评判维度'] == "流畅性":
            test_input = f"模型创作:{test_item['待评判内容']}"
            prompt= get_prompt2(test_input)
            label.append(int(predict_qwen2(prompt,qwen2_model,qwen2_tokenizer)))
        else:
            test_input = f"创作要求:{test_item['创作要求']}。模型创作:{test_item['待评判内容']}"
            prompt= get_prompt3(test_input)
            label.append(int(predict_qwen2(prompt,qwen2_model,qwen2_tokenizer)))
    
    test_df['预测分数'] = label
    submit = test_df[['数据编号', '评判维度', '预测分数']]
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    submit.to_csv(path2, index=False)
    del model
    del tokenizer

def main3():   
    internlm2_5_model_path = '/data/user_data/model/internlm2_5-20b-chat'
    internlm2_5_lora_path = '/data/user_data/lora/internlm2_5_20B_chat_1500_lora'
    internlm2_5_model,internlm2_5_tokenizer = load_model(internlm2_5_model_path,internlm2_5_lora_path)
    label = []
    for i in tqdm(range(len(test_df))):
        test_item = test_df.loc[i]
        if test_item['评判维度'] == "选择题":
            test_input = f"{test_item['创作要求']}。{test_item['待评判内容']}"
            prompt= get_prompt1(test_input)
            label.append(int(predict_internlm2_5(prompt,internlm2_5_model,internlm2_5_tokenizer)))
        elif test_item['评判维度'] == "流畅性":
            test_input = f"{test_item['创作要求']}。{test_item['待评判内容']}"
            prompt= get_prompt2(test_input)
            label.append(int(predict_internlm2_5(prompt,internlm2_5_model,internlm2_5_tokenizer)))
        else:
            test_input = f"{test_item['创作要求']}。{test_item['待评判内容']}"
            prompt= get_prompt3(test_input)
            label.append(int(predict_internlm2_5(prompt,internlm2_5_model,internlm2_5_tokenizer)))
    
    test_df['预测分数'] = label
    submit = test_df[['数据编号', '评判维度', '预测分数']]
    os.makedirs(os.path.dirname(path3), exist_ok=True)
    submit.to_csv(path3, index=False)
    del model
    del tokenizer

def main4():   
    gemma2_model_path = '/data/user_data/model/gemma-2-9b-it'
    gemma2_lora_path = '/data/user_data/lora/gemma-2-9b-it_review_1000_lora'
    gemma2_model,gemma2_tokenizer = load_model(gemma2_model_path,gemma2_lora_path)
    label = []
    for i in tqdm(range(len(test_df))):
        test_item = test_df.loc[i]
        if test_item['评判维度'] == "选择题":
            test_input = f"{test_item['创作要求']}。{test_item['待评判内容']}"
            prompt= get_prompt1(test_input)
            label.append(int(predict_gemma2(prompt,gemma2_model,gemma2_tokenizer)))
        elif test_item['评判维度'] == "流畅性":
            test_input = f"模型创作:{test_item['待评判内容']}"
            prompt= get_prompt2(test_input)
            label.append(int(predict_gemma2(prompt,gemma2_model,gemma2_tokenizer)))
        else:
            test_input = f"创作要求:{test_item['创作要求']}。模型创作:{test_item['待评判内容']}"
            prompt= get_prompt3(test_input)
            label.append(int(predict_gemma2(prompt,gemma2_model,gemma2_tokenizer)))
    
    test_df['预测分数'] = label
    submit = test_df[['数据编号', '评判维度', '预测分数']]
    os.makedirs(os.path.dirname(path4), exist_ok=True)
    submit.to_csv(path4, index=False)
    del model
    del tokenizer

def voted_results(path1,path2,path3,path4):
    gemma2_df = pd.read_csv(path1,encoding='utf-8', encoding_errors='ignore')
    glm4_df = pd.read_csv(path2,encoding='utf-8', encoding_errors='ignore')
    internlm2_5_df = pd.read_csv(path3,encoding='utf-8', encoding_errors='ignore')
    qwen2_df = pd.read_csv(path4,encoding='utf-8', encoding_errors='ignore')
    
    label = []
    for i in tqdm(range(len(glm4_df))):
        gemma2 = gemma2_df.iloc[i]
        glm4 = glm4_df.iloc[i]
        internlm2_5 = internlm2_5_df.iloc[i]
        qwen2 = qwen2_df.iloc[i]
        
        # For multiple-choice questions, use glm4's score
        if glm4['评判维度'] == "选择题":
            label.append(glm4['预测分数'])
        else:
            # Collect the scores for voting
            votes = [gemma2['预测分数'], glm4['预测分数'], internlm2_5['预测分数'], qwen2['预测分数']]
            
            # Count the votes
            vote_count = Counter(votes)
            most_common = vote_count.most_common()
            
            # Check for a clear majority
            if most_common[0][1] >= 3:
                # If a score has 3 or 4 votes, use it
                label.append(most_common[0][0])
            elif most_common[0][1] == 2:
                # Check if there's a tie between two scores
                if len(most_common) > 1 and most_common[1][1] == 2:
                    # Tie detected, default to glm4's score
                    label.append(glm4['预测分数'])
                else:
                    # One score with 2 votes, use it
                    label.append(most_common[0][0])
            else:
                # No majority, default to glm4's score
                label.append(glm4['预测分数'])
    glm4_df['预测分数'] = label
    os.makedirs(os.path.dirname(output), exist_ok=True)
    glm4_df.to_csv(output, index=False)

    

if __name__ == "__main__":
    
    path1 = '/data/user_data/glm4_9b_1000.csv'
    path2 = '/data/user_data/qwen2_7b_1000.csv'
    path3 = '/data/user_data/internlm2.5_20b_1500.csv'
    path4 = '/data/user_data/gemma2_9b_1000.csv'
    output = '/data/prediction_result/result.csv'
    test_df = pd.read_excel('/data/raw_data/test-B.xlsx', engine='openpyxl')
    main1()
    print("glm4完成")
    torch.cuda.empty_cache()
    main2()
    print("qwen2完成")
    torch.cuda.empty_cache()
    main3()
    print("internlm2_5完成")
    torch.cuda.empty_cache()
    main4()
    print("gemma2完成")
    torch.cuda.empty_cache()
    voted_results(path1,path2,path3,path4)
    
