from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated
import operator
import sys

# 初始化模型
myllm = ChatOpenAI(
    api_key="",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="glm-4-flash",
    temperature=0.7,
    max_tokens=2048,
    streaming=True,  # 开启流式模式（关键！）
)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def call_model(state: AgentState):
    """流式调用 LLM"""
    # 使用 stream 而不是 invoke
    response = myllm.stream(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
agent = workflow.compile()

system_msg = SystemMessage(
    content="你是一个历史学家，专门研究德国历史。请用专业但易懂的方式回答用户的问题。"
)


def chat_loop():
    print("=" * 50)
    print("🤖 德国历史专家 AI 已启动（流式输出模式）")
    print("输入 'quit'、'exit' 或 '结束' 退出对话")
    print("=" * 50)

    messages = [system_msg]

    while True:
        user_input = input("\n👤 你: ").strip()

        if user_input.lower() in ["quit", "exit", "结束", "q", "退出"]:
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            print("\n🤖 AI: ", end="", flush=True)  # 不换行，准备流式输出

            # 收集完整回复用于历史记录
            full_content = ""

            # 使用 stream 方法逐 chunk 输出
            for chunk in myllm.stream(messages):# 这里调用了模型的 stream 方法，传入当前的消息历史（messages）。这个方法会返回一个生成器，每次迭代都会得到模型生成的一个新的 chunk（部分回复）。
                # 获取当前 chunk 的内容
                content = chunk.content
                if content:
                    print(content, end="", flush=True)  # 立即输出，不换行
                    full_content += content
            # 这里做的是将用户的输入打印出来，作为对话记录，并且在模型生成回复的过程中，边生成边打印，
            # 最后将完整的回复内容保存在 full_content 变量中，以便后续加入历史记录。

            print()  # 最后换行

            # 将完整回复加入历史
            messages.append(AIMessage(content=full_content))

            # 历史截断
            if len(messages) > 10:
                messages = [system_msg] + messages[-9:]

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            continue


if __name__ == "__main__":
    chat_loop()
