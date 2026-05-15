from langchain_openai import ChatOpenAI
import os
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

myllm = ChatOpenAI(
    api_key="c8ff5d6d099742b0855805b4e59245c4.1ORR5pA9RqwR5wcq",
    base_url="https://open.bigmodel.cn/api/paas/v4",  # 智谱 OpenAI 兼容端点
    model="glm-4-flash",
    temperature=0.7,  # 控制生成文本的随机程度，值越高越随机
    max_tokens=2048,  # 控制生成文本的最大长度
)

agent = create_agent(myllm)


system_meg = SystemMessage(content="你是一个历史学家，专门研究德国历史。")
human_meg = HumanMessage(content="德国皇帝头衔全称是什么？")
response = agent.invoke({"messages": [system_meg, human_meg]})

print(response["messages"][0].content)
print(response["messages"][-1].content)
