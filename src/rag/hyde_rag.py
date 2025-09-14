# coding:utf-8
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
import dashscope
from dashscope import Generation
import os

# 1. 设置Key（请替换成你的实际API Key）
dashscope.api_key = "sk-1bc48ff36"

# 2. 加载嵌入模型（用于文本转向量）
embed_model = SentenceTransformer(r'C:\Users\jys\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620')  # 一个优秀的中文嵌入模型

# 3. 假设我们有一个简单的知识库文档（实际应用中应从文件加载）
knowledge_base = [
    "牛顿第一定律，又称为惯性定律，指出：任何物体在没有外力作用时，总保持匀速直线运动状态或静止状态。",
    "牛顿第二定律指出，物体的加速度与所受合外力成正比，与质量成反比，公式为 F=ma。",
    "牛顿第三定律，又称作用与反作用定律，指出两个物体之间的作用力和反作用力总是大小相等，方向相反，作用在同一直线上。",
    "爱因斯坦的质能方程是 E=mc²，其中E代表能量，m代表质量，c代表光速。",
    "深度学习是机器学习的一个分支，它使用名为深度神经网络的模型。",
]

# 为知识库生成向量并构建Faiss索引
knowledge_vectors = embed_model.encode(knowledge_base)
dimension = knowledge_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(knowledge_vectors.astype('float32'))


# 4. 定义HyDE生成函数（使用Qwen）
def generate_hyde_query(original_query: str) -> str:
    """
    使用Qwen根据用户问题生成一个假设性的答案。
    这个答案可能不准确，但其表述方式更接近知识库中的文本。
    """
    prompt = f"""请根据以下问题，生成一个假设性的、详细的答案。即使你不确定正确答案，也请模仿百科知识的风格和语气来写。问题：{original_query}假设性答案："""

    response = Generation.call(
        model='qwen-plus-2025-04-28',
        prompt=prompt,
        seed=12345,
        top_p=0.8
    )

    hyde_text = response.output['text'].strip()
    print(f"原始查询: {original_query}")
    print(f"HyDE生成: {hyde_text}")

    return hyde_text


# 5. 定义检索函数
def retrieve_with_hyde(user_query: str, top_k: int = 3) -> List[str]:
    """
    1. 使用HyDE生成假设答案。
    2. 将假设答案编码为向量。
    3. 用该向量在Faiss中检索最相似的文档。
    """
    # 生成HyDE查询
    hyde_query = generate_hyde_query(user_query)

    # 将HyDE查询编码为向量
    query_vector = embed_model.encode([hyde_query])

    # 在Faiss中搜索
    distances, indices = index.search(query_vector.astype('float32'), top_k)

    # 返回检索到的文本
    retrieved_docs = [knowledge_base[i] for i in indices[0]]

    return retrieved_docs


# 6. 定义最终答案生成函数（使用Qwen）
def generate_final_answer(user_query: str, contexts: List[str]) -> str:
    """
    将用户查询和检索到的上下文组合成Prompt，让Qwen生成最终答案。
    """
    context_str = "\n".join([f"- {doc}" for doc in contexts])

    prompt = f"""请根据以下提供的上下文信息，回答用户的问题。如果上下文信息不包含答案，请直接说你不知道。上下文信息：{context_str}用户问题：{user_query}请直接给出答案："""

    response = Generation.call(
        model='qwen-max',
        prompt=prompt,
        seed=12345,
        top_p=0.8
    )

    final_answer = response.output['text'].strip()

    return final_answer


# 7. 主流程：完整的RAG with HyDE
def rag_with_hyde(user_query: str):
    # 第一步：通过HyDE检索相关文档
    retrieved_docs = retrieve_with_hyde(user_query)

    print("\n检索到的相关文档：")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i + 1}. {doc}")

    # 第二步：合成最终答案
    final_answer = generate_final_answer(user_query, retrieved_docs)

    print(f"\n最终答案：\n{final_answer}")


# 8. 测试
if __name__ == "__main__":
    user_question = "牛顿第一定律是什么？"
    rag_with_hyde(user_question)

"""
原始查询: 牛顿第一定律是什么？
HyDE生成: **牛顿第一定律**，又称为**惯性定律**，是经典力学的三大基础定律之一，由英国科学家艾萨克·牛顿在其1687年出版的著作《自然哲学的数学原理》中提出。该定律的内容为：

> **“任何物体都会保持静止或匀速直线运动状态，除非有外力迫使它改变这种状态。”**

换句话说，如果一个物体不受外力作用，那么它原有的运动状态（静止或以恒定速度沿直线运动）将不会发生变化。

### 定律的详细解释：

- **惯性**：牛顿第一定律的核心概念是惯性。惯性是指物体抵抗其运动状态发生改变的性质。物体的质量越大，其惯性也越大，也就越难改变它的运动状态。
  
- **理想状态**：在现实世界中，完全不受外力的物体几乎不存在（如摩擦力、空气阻力、重力等总是存在），但该定律提供了一个理论基础，帮助我们理解在没有外力作用下物体的运动行为。

- **参考系的重要性**：牛顿第一定律仅在**惯性参考系**中成立。惯性参考系是指那些相对于宇宙中“固定”点以恒定速度运动或静止的参考系。例如，在地面上观察一个滑块在光滑冰面上滑行时，如果忽略摩擦和空气阻力，滑块将保持匀速直线运动，这与牛顿第一定律一致。

### 历史背景：

虽然牛顿最终将其列为第一定律，但惯性概念的发展可以追溯到伽利略·伽利莱和勒内·笛卡尔的研究。伽利略通过斜面实验得出物体在无外力作用下会持续运动的结论，这为牛顿的惯性定律奠定了基础。

### 实际应用：

牛顿第一定律在日常生活和工程中有广泛应用，例如：

- 汽车安全设计中使用安全带和气囊，是为了在车辆突然停止时防止乘客因惯性继续向前运动而受伤；
- 航天器在太空中关闭引擎后仍能保持高速飞行，正是由于惯性；
- 滑冰运动员在冰面上滑行时，若停止蹬冰，会因摩擦力极小而继续滑行较远距离。

### 总结：

牛顿第一定律不仅是经典力学体系的基石之一，也帮助人类建立了对物体运动本质的深刻理解。它揭示了物体运动状态的自然倾向，并为后续的牛顿第二定律（力与加速度的关系）和第三定律（作用与反作用）提供了逻辑起点。

检索到的相关文档：
1. 牛顿第一定律，又称为惯性定律，指出：任何物体在没有外力作用时，总保持匀速直线运动状态或静止状态。
2. 牛顿第三定律，又称作用与反作用定律，指出两个物体之间的作用力和反作用力总是大小相等，方向相反，作用在同一直线上。
3. 牛顿第二定律指出，物体的加速度与所受合外力成正比，与质量成反比，公式为 F=ma。

最终答案：
牛顿第一定律，又称为惯性定律，指出：任何物体在没有外力作用时，总保持匀速直线运动状态或静止状态。



"""