# coding:utf-8
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import json
import os

# 初始化嵌入模型
embedding_model = SentenceTransformer(r'C:\Users\jys\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620')

# Qwen API配置
QWEN_API_KEY = "sk-1bc48ff3606"  # 替换为您的实际API密钥
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def call_qwen(prompt):
    """调用Qwen API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}"
    }
    payload = {
        "model": "qwen-turbo",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        "parameters": {
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['output']['text']
    except Exception as e:
        print(f"API调用失败: {e}")
        return None


# 1. 准备文档数据
documents = [
    "员工报销需要提供发票和审批单，15个工作日内完成报销。",
    "请假需提前在OA系统申请，紧急情况可事后补办手续。",
    "密码必须包含字母、数字和特殊字符，且长度至少8位。",
    "新产品发布流程包括需求评审、设计、开发、测试和发布五个阶段。"
]

# 2. 生成查询问题 (Doc2Query)
print("为文档生成查询问题...")
doc_queries = []
for doc in documents:
    prompt = f"请为以下文本生成3个用户可能会提出的问题:\n\n文本: {doc}\n\n生成的问题:"
    response = call_qwen(prompt)

    if response:
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        doc_queries.append(queries[:3])
        print(f"文档: {doc[:20]}...")
        print(f"生成的问题: {queries[:3]}")
    else:
        # 如果API调用失败，使用简单的问题
        doc_queries.append([f"关于{doc[:10]}...", f"如何{doc[:10]}...", f"{doc[:10]}有什么要求..."])
    print()

# 3. 创建FAISS索引
# 合并文档和生成的问题
all_texts = documents.copy()
for queries in doc_queries:
    all_texts.extend(queries)

# 生成嵌入向量
embeddings = embedding_model.encode(all_texts)

# 创建FAISS索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))


# 4. 查询改写函数 (Query2Doc)
def rewrite_query(query):
    prompt = f"请根据以下问题生成一段详细的答案文档:\n\n问题: {query}\n\n生成的答案文档:"
    response = call_qwen(prompt)
    return response if response else query


# 5. 检索函数
def search(query):
    print(f"原始查询: {query}")

    # 策略1: 直接检索
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), 3)

    print("直接检索结果:")
    for i, idx in enumerate(indices[0]):
        if idx < len(all_texts):
            print(f"  {i + 1}. {all_texts[idx]}")

    # 策略2: 查询改写后检索
    expanded_query = rewrite_query(query)
    if expanded_query != query:
        print(f"改写后的查询: {expanded_query}")
        expanded_embedding = embedding_model.encode([expanded_query])
        distances, indices = index.search(np.array(expanded_embedding).astype('float32'), 3)

        print("改写后检索结果:")
        for i, idx in enumerate(indices[0]):
            if idx < len(all_texts):
                print(f"  {i + 1}. {all_texts[idx]}")

    print("-" * 50)


# 6. 测试查询
queries = ["怎么报销", "如何请假", "密码要求", "发布流程"]
for query in queries:
    search(query)

"""
为文档生成查询问题...
文档: 员工报销需要提供发票和审批单，15个工作...
生成的问题: ['1. 员工报销需要哪些材料？', '2. 报销流程需要多长时间？', '3. 如果超过15个工作日还没报销，应该怎么办？']

文档: 请假需提前在OA系统申请，紧急情况可事后...
生成的问题: ['1. 请假流程是怎样的？', '2. 如果遇到紧急情况，是否可以先请假再补办手续？', '3. OA系统申请请假有什么具体要求吗？']

文档: 密码必须包含字母、数字和特殊字符，且长度...
生成的问题: ['1. 密码需要满足哪些要求？', '2. 什么样的字符可以算作特殊字符？', '3. 密码长度最少是多少位？']

文档: 新产品发布流程包括需求评审、设计、开发、...
生成的问题: ['1. 新产品发布流程的具体步骤有哪些？', '2. 需求评审在新产品发布中起到什么作用？', '3. 测试阶段在新产品发布流程中为什么很重要？']

原始查询: 怎么报销
直接检索结果:
  1. 2. 报销流程需要多长时间？
  2. 1. 员工报销需要哪些材料？
  3. 3. 如果超过15个工作日还没报销，应该怎么办？
改写后的查询: **报销流程说明文档**

一、报销概述  
报销是指员工在因公出差、业务往来或其他公司规定范围内产生的合理费用，经审批后向公司财务部门申请资金返还的过程。为确保报销流程的规范性和透明度，公司制定了统一的报销制度和操作流程。

二、适用范围  
本流程适用于公司所有正式员工，适用于以下情况：  
- 因公出差所产生的交通、住宿、餐饮等费用；  
- 业务招待费用；  
- 公司采购或服务相关的支出；  
- 其他经批准的业务相关费用。

三、报销前准备  
1. **保留有效票据**  
   所有报销项目必须提供合法、有效的原始发票或收据，如：  
   - 机票、火车票、出租车发票等交通票据；  
   - 酒店发票、餐饮发票等消费凭证；  
   - 服务合同、付款凭证等业务相关票据。  
   *注：发票内容需与实际支出相符，且在有效期内。*

2. **填写报销单**  
   在公司内部系统（如OA系统）中填写《费用报销单》，包括以下信息：  
   - 报销人姓名、部门、岗位；  
   - 报销日期、费用类型（如差旅费、业务招待费等）；  
   - 费用明细（包括金额、用途、发生时间、地点等）；  
   - 附件清单（如发票、合同、审批单等）。

3. **提交审批**  
   报销单填写完成后，需按照公司规定的审批流程提交至相关负责人审批。  
   - 一般情况下，需先由直属上级审批；  
   - 若涉及大额费用或特殊事项，还需提交至财务部或更高管理层审批。

四、报销流程步骤  
1. **提交报销申请**  
   在公司指定的报销系统中填写并提交报销单，并上传相关票据的扫描件或照片。

2. **部门审批**  
   直属上级根据实际情况对报销内容进行审核，确认是否符合公司政策及预算要求。

3. **财务审核**  
   财务部门对报销单及附件进行合规性审查，确保票据真实、金额准确、用途合理。

4. **报销支付**  
   审核通过后，财务部门将在规定时间内将报销款项支付至员工指定账户（如工资卡或企业微信等）。

五、注意事项  
1. 报销必须在费用发生后的**30日内**完成，逾期不予受理。  
2. 禁止虚报、假报、重复报销等违规行为，一经发现将按公司相关规定处理。  
3. 报销过程中如遇到问题，可联系财务部或所属部门主管咨询。  
4. 每月月底前，财务部会汇总当月报销情况，并进行账务处理。

六、常见问题解答  
Q1：没有发票可以报销吗？  
A：原则上必须提供正规发票，特殊情况需提前报备并附相关证明材料。

Q2：报销金额超过多少需要特别审批？  
A：具体标准根据公司财务制度而定，一般超过5000元需总经理审批。

Q3：报销后多久能到账？  
A：通常在审批通过后3-7个工作日内到账，具体时间视财务安排而定。

七、联系方式  
如有任何报销相关问题，请联系：  
- 财务部：分机号：8001，邮箱：finance@company.com  
- OA系统技术支持：分机号：8002

---

**备注：** 本流程依据公司现行财务制度制定，如有变动以最新通知为准。
改写后检索结果:
  1. 员工报销需要提供发票和审批单，15个工作日内完成报销。
  2. 2. 报销流程需要多长时间？
  3. 1. 员工报销需要哪些材料？
--------------------------------------------------
原始查询: 如何请假
直接检索结果:
  1. 1. 请假流程是怎样的？
  2. 2. 如果遇到紧急情况，是否可以先请假再补办手续？
  3. 3. OA系统申请请假有什么具体要求吗？
改写后的查询: **如何请假**

请假是指员工因个人原因（如疾病、家庭事务、培训、会议或其他特殊情况）需要暂时离开工作岗位的行为。为了确保工作正常进行，同时保障员工的合法权益，公司通常会制定明确的请假制度。以下是一般情况下请假的流程和注意事项：

---

### 一、了解公司的请假制度

在请假前，员工应首先查阅公司的人事管理制度或员工手册，了解以下内容：

- **请假类型**：包括事假、病假、年假、婚假、产假、丧假、调休等。
- **请假时长限制**：不同类型的假期有不同的天数规定。
- **审批权限**：根据请假时间长短，可能需要不同层级的领导审批。
- **请假申请方式**：是否需通过书面、电子邮件、OA系统或其他平台提交。

---

### 二、提前申请

请假应当**提前申请**，尤其是较长时间的假期。一般建议如下：

- **事假**：提前1-3个工作日申请，特殊情况可事后补办手续。
- **病假**：如因突发疾病无法上班，应在当天及时通知直属领导，并在恢复后补交相关证明。
- **年假**：应提前与主管沟通并安排好工作交接，避免影响团队进度。

---

### 三、填写请假申请表

大多数公司要求员工填写《请假申请表》或通过内部系统提交请假申请。申请内容通常包括：

- 姓名
- 部门
- 职位
- 请假类型
- 请假起止日期
- 请假事由
- 联系方式（如有需要）
- 直属领导审批意见

---

### 四、提交审批

根据公司规定，将请假申请提交给相应的审批人，通常是：

- **直属上级**：负责初步审核请假合理性。
- **部门负责人**：对请假进行最终审批。
- **人力资源部**：记录请假信息并更新考勤系统。

部分公司还可能要求提供相关证明材料，例如：

- 病假需提供医院出具的诊断证明；
- 婚假、产假需提供结婚证、出生证明等。

---

### 五、工作交接

在请假前，员工应做好工作交接，确保不影响团队的正常运作。具体包括：

- 向同事或接替人员说明当前的工作任务和进度；
- 提供必要的文件资料或操作指引；
- 保持通讯畅通，以便紧急情况联系。

---

### 六、请假期间的注意事项

- 请假期间应遵守公司相关规定，不得从事与本职工作冲突的活动。
- 如需延长请假时间，应提前申请并获得批准。
- 未经批准擅自缺勤，可能会被视为旷工，影响考勤记录和绩效考核。

---

### 七、销假与返岗

请假结束后，员工应在**返岗当日**办理销假手续，向直属领导或HR部门报到。如因特殊情况未能按时返岗，应及时联系并说明原因。

---

### 八、常见问题解答

**Q1：如果临时有急事，能否不提前请假？**  
A：原则上应提前请假，但如遇特殊情况，应及时电话或微信告知主管，并在事后补办手续。

**Q2：请假是否会影响工资？**  
A：视请假类型而定。事假通常无薪，病假按公司政策处理，年假则不影响工资。

**Q3：是否可以跨部门请假？**  
A：一般需经所在部门领导审批，如涉及其他部门协调，需提前沟通。

---

### 九、总结

请假是员工合理安排个人事务的重要方式，但必须遵循公司规定，做到**提前申请、合理安排、妥善交接**。良好的请假制度有助于维护企业正常运营，同时也体现了对员工权益的尊重。

如对请假制度有任何疑问，建议咨询人力资源部门或直接与直属领导沟通。
改写后检索结果:
  1. 1. 请假流程是怎样的？
  2. 请假需提前在OA系统申请，紧急情况可事后补办手续。
  3. 3. OA系统申请请假有什么具体要求吗？
--------------------------------------------------
原始查询: 密码要求
直接检索结果:
  1. 1. 密码需要满足哪些要求？
  2. 密码必须包含字母、数字和特殊字符，且长度至少8位。
  3. 3. 密码长度最少是多少位？
Traceback (most recent call last):
  File "D:\pycharm\PyCharm 2022.2.4\plugins\python\helpers\pydev\pydevd.py", line 1496, in _exec
    pydev_imports.execfile(file, globals, locals)  # execute the script
  File "D:\pycharm\PyCharm 2022.2.4\plugins\python\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "D:\code\rag_learning\query2Doc_Doc2Query.py", line 126, in <module>
    search(query)
  File "D:\code\rag_learning\query2Doc_Doc2Query.py", line 111, in search
    print(f"改写后的查询: {expanded_query}")
UnicodeEncodeError: 'gbk' codec can't encode character '\u2705' in position 409: illegal multibyte sequence
python-BaseException

"""