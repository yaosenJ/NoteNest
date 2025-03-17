# graphrag��Ŀѧϰ

## 1.������װ

```bash
conda create -n GraphRAG python=3.11
conda activate GraphRAG 
pip install graphrag

git clone https://github.com/microsoft/graphrag.git
mkdir -p .\ragtest\input
python -m graphrag init --root .\ragtest
```
���������������Ŀ¼��ʽ���£�

![](./img/directory.png)

��**settings.yaml**�У� ���model��api_base��Ӧ������ֵ������ֱ��ģʽ������Ҫapi_base
��**env**�ļ��У������ʹ��ģ�͵�api_key

## 2.������������
```bash
python -m graphrag index --root .\ragtest
```

����������־
![](./img/log1.png)
![](./img/log2.png)
![](./img/log3.png)
![](./img/log4.png)

## 3.graphrag�ʴ�

```bash
graphrag query --root .\ragtest --method local --query "ɽ�����ؾ��Ͽ�ҵ�������ι�˾�����¹�ԭ����ʲô"
graphrag query --root .\ragtest --method global --query "ɽ�����ؾ��Ͽ�ҵ�������ι�˾�����¹�ԭ����ʲô"
```
![](./img/log5.png)
![](./img/log6.png)

## ����
![](./img/error1.png)

�����ʽ������txt�ļ���ʹ��utf-8����

![](./img/error2.png)

�����ʽ�� ��settings.yaml�У����:  **encoding_model: cl100k_base**

openai.BadRequestError: Error code: 400 - {'error': {'code': 'InvalidParameter', 'param': None, 'message': '<400> InternalError.Algo.InvalidParameter: Value error, batch size is invalid, it should not be larger than 10.: input.contents', 'type': 'InvalidParameter'}, 'id': 'bcac04a7-19f7-94cf-a09e-3464ee6b6bd2', 'request_id': 'bcac04a7-19f7-94cf-a09e-3464ee6b6bd2'}
22:32:14,959 graphrag.callbacks.file_workflow_callbacks INFO Error running pipeline! details=None
22:32:14,977 httpx INFO HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 400 Bad Request"

��settings.yaml�У�����concurrent_requests��ֵ�����߻�ģ��
