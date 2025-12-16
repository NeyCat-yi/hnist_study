# Agent 学习

## 一个例子

这是一个简单的可运行例子，当然，需要在本地cmd运行：ollama run qwen2.5:7b

```python
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="你好！")
```



## Memory

多种记忆方式

```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

from langchain.memory import ConversationBufferWindowMemory # 导入窗口记忆(区别于普通记忆)
memory = ConversationBufferWindowMemory(k=1) # 只保留最近1轮对话

from langchain.memory import ConversationTokenBufferMemory # 导入基于Token的记忆
memory = ConversationTokenBufferMemory(max_token_limit=50) # 设置最大Token数为50

from langchain.memory import ConversationSummaryBufferMemory # 基于token总数的记忆
memory = ConversationSummaryBufferMemory(max_token_limit=50) # 
```



# 一个Search Agent

## 导包

```python
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
```

## 初始化本地大模型

```python
# 1. 初始化本地的大模型 (Qwen 2.5)
# 注意：确保你本地的 Ollama 已经启动，并且通过 'ollama pull qwen2.5:7b' 下载了模型
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0  # 设置为0，让模型回答更严谨，减少幻觉
)
```

## 定义工具 Tools

```python
# 这里我们使用 DuckDuckGo 搜索，它不需要 API Key，适合练手
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
```

## 定义 Prompt

```python
# 3. 定义 Prompt (提示词模板)
# 这是一个标准的 ReAct 模板，告诉模型如何思考、如何使用工具
template = '''
尽可能回答以下问题。你可以使用以下工具：

{tools}

使用以下格式：

Question: 你必须回答的输入问题
Thought: 你应该总是思考该做什么
 Action Input: 动作的输入（例如搜索关键词）
Observation: 动作的结果
... (Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 针对原始问题的最终答案

开始！

Question: {input}
Thought:{agent_scratchpad}
'''

prompt = PromptTemplate.from_template(template)
```

## 创建 Agent

```python
# 4. 创建 Agent
# 将 LLM、工具和提示词结合在一起
agent = create_react_agent(llm, tools, prompt)
```

## 创建执行器

```python
# 5. 创建执行器 (AgentExecutor)
# AgentExecutor 负责运行 Agent 的循环（思考 -> 执行 -> 再思考）
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,       # 开启详细模式，可以在控制台看到思考过程
    handle_parsing_errors=True # 容错处理，防止模型输出格式微小错误导致崩溃
)
```

## 运行测试

```python
# 6. 运行测试
if __name__ == "__main__":
    print("Agent 已启动... (输入 'quit' 退出)")
    while True:
        user_input = input("\n请输入你想查询的问题: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        try:
            # invoke 触发 Agent 运行
            response = agent_executor.invoke({"input": user_input})
            print(f"\n========\n最终回答: {response['output']}\n========")
        except Exception as e:
            print(f"发生错误: {e}")
```

