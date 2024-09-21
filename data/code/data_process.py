import pandas as pd
import os
import json

def convert_to_builtin_type(obj):
    if isinstance(obj, np.int64):  # 如果对象是 int64 类型
        return int(obj)  # 转换为内置的 int 类型
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def data_process(input_path,output_path):
    train_df = pd.read_csv(input_path, encoding='gb2312', encoding_errors='ignore')
    res = []
    for i in range(len(train_df)):
    # for i in range(10):
        llm_item = train_df.loc[i]
        if (llm_item['评判维度']=="选择题"):
            tmp =  {
            "instruction": "如果模型回复与正确选项一致，则输出1；如果不一致，则输出0。",
            "input": f"{llm_item['创作要求']}。{llm_item['待评判内容']}",
            "output": str(llm_item['标注分值']) 
            }
            res.append(tmp)
    
        elif (llm_item['评判维度']=="流畅性"):
            tmp =   {
            "instruction": """下面是一个模型完成的创作内容。按照流畅性评分标准给模型创作打分(只取1分、2分、3分、4分、5分其一)。流畅性评分标准如下：
            1分: 非常不流畅，不具备可读性，语法错误明显，难以理解，大量拼写错误和错别字，影响阅读，表达不清晰，难以捉摸要表达的意思，每百字平均错误数超过2.5个。
            2分: 具有可读性，但较不流畅，常见语法错误多，需花费一定时间理解，一些拼写错误和错别字，阅读中断，表达较为模糊，需用一些猜测才能明白含义，每百字平均错误数介于2至2.5个。
            3分：基本流畅，存在少量语法错误，但影响较小，稍有拼写错误，但不影响阅读，主要意思表达清楚，但部分地方表述不够准确，每百字平均错误数介于1至2个。
            4分：较流畅，语法错误稀少，易读性较高，几乎无拼写错误，阅读顺畅，表达清晰、准确，容易理解，每百字平均错误数介于0.5至1个。
            5分：非常流畅，语法、拼写完美，阅读体验优秀，表达精炼、准确、得体；文句优美，行文连贯，思维严密，每百字平均错误数在0至0.5个之间。""",
            "input": f" 模型创作:{llm_item['待评判内容']}",
            "output": str(llm_item['标注分值']) 
            }
            res.append(tmp)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, mode='w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4, default=convert_to_builtin_type)