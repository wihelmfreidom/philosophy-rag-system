from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    api_key="c8ff5d6d099742b0855805b4e59245c4.1ORR5pA9RqwR5wcq",
    base_url="https://open.bigmodel.cn/api/paas/v4",  # 智谱 OpenAI 兼容端点
    model="glm-4-flash",
)
response = llm.invoke(
    "“独化”的概念出自《齐物论》“罔两问景”一段的注解，这一段注解的原文是什么呢（世或谓罔两待景，景待形，形待造物者。补全原文）.用中文回答"
)
print("模型响应:", response.content)
# # 打印模型配置确认
# print(f"当前使用模型: {llm.model_name}")  # 应该是 glm-4-flash
# print(f"Base URL: {llm.openai_api_base}")  # 应该是智谱的地址

# 1. 简单文本输入
# response = llm.invoke("什么是人工智能？用中文回答")
# print("模型响应:", response.content)

# 2. 对话输入
# conversation = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant that translates English to French.",
#     },
#     {"role": "user", "content": "Translate: I love programming."},
#     {"role": "assistant", "content": "J'adore la programmation."},
#     {"role": "user", "content": "Translate: I love building applications."},
# ]

# response = llm.invoke(conversation)
# print(response.content)  # AIMessage("J'adore créer des applications.")


# 3. 流式输出,就是边生成边输出，适合长文本或需要实时反馈的场景
# full = None  # None | AIMessageChunk
# for chunk in llm.stream("什么是流体力学？用中文回答,详细说明"):
#     full = chunk if full is None else full + chunk
#     # print(full.text)


# print(full.content_blocks)  # 输出完整内容

# # 4. 批量输入,适合同时处理多个请求的场景，可以提高效率
# for response in llm.batch_as_completed(
#     [
#         "为什么人要吃饭？",
#         "人工智能是什么？",
#         "什么是海洋科学？",
#     ]
# ):
#     # print(response)
#     print("-" * 50)
#     print(type(response))  # tuple (response, index)
#     print(f"输入索引: {response[0]}")  # 输入的索引
#     print(f"模型响应: {response[1].content}")  # 模型的响应内容
