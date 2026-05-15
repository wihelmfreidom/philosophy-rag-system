# from typing import Literal
# from langgraph.graph import StateGraph, START, END
# from langgraph.types import interrupt, Command, RetryPolicy
# from langchain_openai import ChatOpenAI
# from langchain.messages import HumanMessage

# llm = ChatOpenAI(model="gpt-5-nano")


# def read_email(state: EmailAgentState) -> dict:
#     # state: EmailAgentState表示这个函数接收一个EmailAgentState类型的参数state，包含了当前状态的信息，比如email_content和sender_email等。
#     #  -> dict表示这个函数返回一个字典类型的结果，这个字典会被用来更新状态或者传递给下一个节点。
#     """Extract and parse email content"""
#     # In production, this would connect to your email service
#     return {
#         "messages": [
#             HumanMessage(content=f"Processing email: {state['email_content']}")
#         ]
#     }


# # HumanMessage(content=f"Processing email: {state['email_content']}")是创建了一个HumanMessage对象（类实例化），
# # 内容是"Processing email: "加上当前状态中的email_content字段的值。这个消息会被传递给LLM进行处理，或者用于后续节点的输入。


# def classify_intent(
#     state: EmailAgentState,
# ) -> Command[
#     Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]
# ]:
#     # -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]的意思是这个函数返回一个Command对象，
#     # 这个Command对象的goto属性只能是"search_documentation"、"human_review"、"draft_response"或"bug_tracking"中的一个，表示下一步要跳转到哪个节点。
#     # Command对象是StateGraph中用于控制流程跳转和状态更新的一个重要类型，它包含了update属性（一个字典，用于更新状态）和goto属性（一个字符串，表示下一步要跳转的节点）。
#     # 通过返回不同的Command对象，可以实现根据分类结果动态跳转到不同的节点。而Literal类型则是Python中的一个类型提示工具，
#     # 用于限制某个变量只能取特定的几个值，这里用来限制goto属性只能是指定的几个节点名称。
#     """Use LLM to classify email intent and urgency, then route accordingly"""

#     # Create structured LLM that returns EmailClassification dict
#     structured_llm = llm.with_structured_output(EmailClassification)

#     # Format the prompt on-demand, not stored in state
#     classification_prompt = f"""
#     Analyze this customer email and classify it:

#     Email: {state['email_content']}
#     From: {state['sender_email']}

#     Provide classification including intent, urgency, topic, and summary.
#     """

#     # Get structured response directly as dict
#     classification = structured_llm.invoke(classification_prompt)

#     # Determine next node based on classification
#     if classification["intent"] == "billing" or classification["urgency"] == "critical":
#         goto = "human_review"
#     elif classification["intent"] in ["question", "feature"]:
#         goto = "search_documentation"
#     elif classification["intent"] == "bug":
#         goto = "bug_tracking"
#     else:
#         goto = "draft_response"

#     # Store classification as a single dict in state
#     return Command(update={"classification": classification}, goto=goto)
#     # 创建了一个Command对象，update属性是一个字典，包含了分类结果的整个字典（classification），
#     # 而不是把分类结果的每个字段都单独存储到state中。goto属性根据分类结果决定下一步要跳转到哪个节点。
#     # update 和 goto 是 Command 对象的两个重要属性，update 用于更新状态，goto 用于控制流程跳转。
#     # 通过返回不同的 Command 对象，可以实现根据分类结果动态跳转到不同的节点，并且将分类结果作为一个整体存储到状态中，供后续节点使用。


# # goto表示下一步要跳转的节点，update表示要更新到state中的内容，这里是把分类结果存储到state中，供后续节点使用。


# def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
#     """Search knowledge base for relevant information"""

#     # Build search query from classification
#     classification = state.get("classification", {})
#     query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

#     try:
#         # Implement your search logic here
#         # Store raw search results, not formatted text
#         search_results = [
#             "Reset password via Settings > Security > Change Password",
#             "Password must be at least 12 characters",
#             "Include uppercase, lowercase, numbers, and symbols",
#         ]
#     except SearchAPIError as e:
#         # For recoverable search errors, store error and continue
#         search_results = [f"Search temporarily unavailable: {str(e)}"]

#     return Command(
#         update={"search_results": search_results},  # Store raw results or error
#         goto="draft_response",
#     )


# def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
#     """Create or update bug tracking ticket"""

#     # Create ticket in your bug tracking system
#     ticket_id = "BUG-12345"  # Would be created via API

#     return Command(
#         update={
#             "search_results": [f"Bug ticket {ticket_id} created"],
#             "current_step": "bug_tracked",
#         },
#         goto="draft_response",
#     )


# def draft_response(
#     state: EmailAgentState,
# ) -> Command[Literal["human_review", "send_reply"]]:
#     """Generate response using context and route based on quality"""

#     classification = state.get("classification", {})

#     # Format context from raw state data on-demand
#     context_sections = []

#     if state.get("search_results"):
#         # Format search results for the prompt
#         formatted_docs = "\n".join([f"- {doc}" for doc in state["search_results"]])
#         context_sections.append(f"Relevant documentation:\n{formatted_docs}")

#     if state.get("customer_history"):
#         # Format customer data for the prompt
#         context_sections.append(
#             f"Customer tier: {state['customer_history'].get('tier', 'standard')}"
#         )

#     # Build the prompt with formatted context
#     draft_prompt = f"""
#     Draft a response to this customer email:
#     {state['email_content']}

#     Email intent: {classification.get('intent', 'unknown')}
#     Urgency level: {classification.get('urgency', 'medium')}

#     {chr(10).join(context_sections)}

#     Guidelines:
#     - Be professional and helpful
#     - Address their specific concern
#     - Use the provided documentation when relevant
#     """

#     response = llm.invoke(draft_prompt)

#     # Determine if human review needed based on urgency and intent
#     needs_review = (
#         classification.get("urgency") in ["high", "critical"]
#         or classification.get("intent") == "complex"
#     )

#     # Route to appropriate next node
#     goto = "human_review" if needs_review else "send_reply"

#     return Command(
#         update={"draft_response": response.content},  # Store only the raw response
#         goto=goto,
#     )


# def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
#     """Pause for human review using interrupt and route based on decision"""

#     classification = state.get("classification", {})

#     # interrupt() must come first - any code before it will re-run on resume
#     human_decision = interrupt(
#         {
#             "email_id": state.get("email_id", ""),
#             "original_email": state.get("email_content", ""),
#             "draft_response": state.get("draft_response", ""),
#             "urgency": classification.get("urgency"),
#             "intent": classification.get("intent"),
#             "action": "Please review and approve/edit this response",
#         }
#     )

#     # Now process the human's decision
#     if human_decision.get("approved"):
#         return Command(
#             update={
#                 "draft_response": human_decision.get(
#                     "edited_response", state.get("draft_response", "")
#                 )
#             },
#             goto="send_reply",
#         )
#     else:
#         # Rejection means human will handle directly
#         return Command(update={}, goto=END)


# def send_reply(state: EmailAgentState) -> dict:
#     """Send the email response"""
#     # Integrate with email service
#     print(f"Sending reply: {state['draft_response'][:100]}...")
#     return {}


# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.types import RetryPolicy

# # Create the graph
# workflow = StateGraph(EmailAgentState)

# # Add nodes with appropriate error handling
# workflow.add_node("read_email", read_email)
# workflow.add_node("classify_intent", classify_intent)

# # Add retry policy for nodes that might have transient failures
# workflow.add_node(
#     "search_documentation",
#     search_documentation,
#     retry_policy=RetryPolicy(max_attempts=3),
# )
# workflow.add_node("bug_tracking", bug_tracking)
# workflow.add_node("draft_response", draft_response)
# workflow.add_node("human_review", human_review)
# workflow.add_node("send_reply", send_reply)

# # Add only the essential edges
# workflow.add_edge(START, "read_email")
# workflow.add_edge("read_email", "classify_intent")
# workflow.add_edge("send_reply", END)
# # 这里添加的边是必须的，START到read_email，read_email到classify_intent，
# # 以及send_reply到END。其他节点之间的跳转由classify_intent根据分类结果动态决定，不需要在这里预先定义边。
# # 换句话说，这是添加到边表示这里的边代表的流程是固定的、每次都要执行的，而不是根据分类结果动态跳转的。
# # Compile with checkpointer for persistence, in case run graph with Local_Server --> Please compile without checkpointer
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# # 这句代码是编译图并使用MemorySaver作为checkpointer，这样可以在运行过程中保存状态，以便在需要时恢复。
# # 根据提示，如果要在本地服务器上运行，可以选择不使用checkpointer，直接编译图即可：
# # app = workflow.compile()

# # 图的结点就是具体的动作的函数，边表示结点之间的流程关系，START和END是特殊的结点，表示流程的开始和结束。
# # 通过add_edge方法添加边，定义流程的顺序。最后通过compile方法编译图，生成一个可执行的应用实例。
