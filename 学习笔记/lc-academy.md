# Module-0

## 📚 课程概述

欢迎来到 LangChain Academy！

### 背景介绍

LangChain 的目标是让构建 LLM 应用变得简单。Agent（代理）是一种可以构建的 LLM 应用。Agent 之所以备受关注，是因为它们可以自动化处理复杂的多步骤任务。

然而，在实践中，构建能够可靠执行这些任务的系统非常困难。通过与用户合作将 Agent 投入生产，我们学到了更多的控制和可观测性对于构建可靠的系统至关重要。

为了解决这个问题，我们构建了 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — 一个用于构建单 Agent 和多 Agent 应用的框架。

### 课程结构

课程由多个模块组成，每个模块专注于 LangGraph 相关的特定主题。每个模块文件夹包含一系列笔记本。

### 准备工作

开始之前，请按照 `README` 中的说明创建环境并安装依赖。

---

## 💬 聊天模型

在本课程中，我们使用聊天模型，它接收消息序列作为输入并返回消息作为输出。LangChain 通过[第三方集成](https://docs.langchain.com/)支持许多模型。

### 设置 API 密钥

```python
import os, getpass

def _set_env(var:  str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}:  ")

# _set_env("OPENAI_API_KEY")
```

### 常见参数

聊天模型有两个最常见的参数：

- **`model`**: 模型名称
- **`temperature`**: 采样温度（控制随机性）
  - 低温度（接近 0）：输出更确定和有针对性，适合需要准确性的任务
  - 高温度：输出更具创意和多样性，适合创意任务

### 初始化模型

```python
from langchain.chat_models import init_chat_model

qwen = init_chat_model(
    model_provider="ollama",
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.7,
)
```

```python
# 这一套兼容性更强
from langchain_ollama import ChatOllama
qwen = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)
```



### 主要方法

- **`stream`**: 流式返回响应的块
- **`invoke`**: 调用链执行输入

### 使用消息

聊天模型接收消息作为输入。消息包含：
- **role**: 描述说话者的角色
- **content**: 消息内容

```python
from langchain_core.messages import HumanMessage

# 创建消息
msg = HumanMessage(content="Hello world", name="Lance")

# 消息列表
messages = [msg]

# 调用模型
qwen.invoke(messages)
```

### 简化用法

可以直接传入字符串，它会自动转换为 `HumanMessage`：

```python
qwen.invoke("hello world")

```

### 模型一致性

所有聊天模型的接口一致，通常在每个笔记本开始时初始化一次。这样可以轻松地在不同提供商之间切换。

---

## 🔍 搜索工具

本课程使用 [Tavily](https://tavily.com/)，这是一个为 LLM 和 RAG 优化的搜索引擎，致力于提供高效、快速和持久的搜索结果。

### 设置 Tavily

```python
_set_env("TAVILY_API_KEY")

from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3)

# 执行搜索
data = tavily_search.invoke({"query": "What is LangGraph?"})
search_docs = data.get("results", data)

# 查看搜索结果
search_docs
```

### 搜索结果示例

返回的搜索结果包含：
- **url**: 来源网址
- **title**: 标题
- **content**: 内容摘要
- **score**: 相关性得分

---



# Module-1

## 导包

```python
import os, getpass # 和配置 key 相关
from langchain_ollama import ChatOllama # 模型相关
from langsmith import traceable # LangSmith 相关
from langgraph.graph import MessagesState # 追加到消息列表 以 MessagesState 作为状态来传递 可以点进去看看样子
from langchain_core.messages import HumanMessage, SystemMessage # 用户信息和系统信息
from langgraph.graph import START, END, StateGraph # 创建 图 相关
from langgraph.prebuilt import tools_condition # 如果 LLM 决定调用工具，通向叫 "tools" 的节点，否则去 END
from langgraph.prebuilt import ToolNode # 内置的 ToolNode 组件，只需传入工具列表即可初始化它，相当于一个节点
from IPython.display import Image, display # 展示相关
from langgraph.checkpoint.memory import MemorySaver # Agent memory 相关
```

## LangSmith

输入API 再给节点和工具加上 @traceable 即可追踪

```python
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```



## 模型

```python
qwen = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)
```

```python
from langchain_openai import ChatOpenAI

# 使用 ChatOpenAI 类，但是指向 DeepSeek 的服务器
llm = ChatOpenAI(
    model="qwen-plus",  # 或者 "deepseek-coder"
    api_key="*",  # 你的 DeepSeek API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 关键点：修改 Base URL
    temperature=0
)
```



## 工具

```python
@traceable # LangSmith 追踪
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
@traceable
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@traceable
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools =  [add, multiply, divide]
```

## 模型绑定工具

把 Python 函数里的 __doc__ 变成 JSON 说明书给 LLM，LLM 通过这个 JSON 来调用工具

```python
llm_with_tools = qwen.bind_tools(tools) # parallel_tool_calls=False 关闭并行工具调用（Ollama 没有这个参数）
```

**展示：**

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
# 魔法就在这里：将函数转换为工具格式
tool_json = convert_to_openai_tool(multiply)

import json
print(json.dumps(tool_json, indent=2, ensure_ascii=False))
```

```python
{
  "type": "function",
  "function": {
    "name": "multiply",
    "description": "计算 a 乘 b 就用这个方法",
    "parameters": {
      "properties": {
        "a": {
          "description": "first int",
          "type": "integer"
        },
        "b": {
          "description": "second int",
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ],
      "type": "object"
    }
  }
}
```

## 构建图



```python
# node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]} # 调用模型时，会自动返回数据类型(AIMessage)

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm) # llm 节点
builder.add_node("tools", ToolNode([multiply])) # 增加一个工具节点

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition, # 要么通向一个 "tools" 节点，要么 通向 END  
)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![屏幕截图 2026-01-05 214248](C:\Users\34356\Desktop\md\resources\屏幕截图 2026-01-05 214248.png)



## Agent memory

LangGraph 可以使用检查点来自动保存每一步之后的图状态。

这个内置的持久化层为我们提供了内存，使 LangGraph 能够从上次状态更新的位置继续执行。

最容易使用的检查点之一是 `MemorySaver`，它是一个用于存储图状态的内存键值存储。

我们只需要使用检查点编译图，我们的图就有了内存！

```python
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

当我们使用内存时，需要指定一个 `thread_id`。

这个 `thread_id` 将存储我们图的状态集合。

以下是一个示意图：

* 检查点在图的每一步写入状态
* 这些检查点保存在一个线程中
* 我们以后可以使用 `thread_id` 访问该线程
* ![state.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e0e9f526b41a4ed9e2d28b_agent-memory2.png)

**建立一个 checkpoint**

```python
# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()
```

**运行时，把 checkpoint 也加入到 messages**

```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
```



# Module-2

## state-reducers

自定义 reducer

例如，我们可以定义自定义 reducer 逻辑来合并列表，并处理输入中一个或两个都为 `None` 的情况

```python
def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]

class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list] 
```

 `MessagesState` 有一个内置的 `messages` 键

 它还有一个内置的 `add_messages` reducer 来处理该键

这两者是等效的。

为了简洁起见，我们将通过 `from langgraph.graph import MessagesState` 来使用 `MessagesState` 类。

```python
# 新增信息
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# New message to add
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
add_messages(initial_messages , new_message)
```

```python
# 根据 id 重写信息
# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                   ]

# New message to add
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

# Test
add_messages(initial_messages , new_message)
```

```python
# 移除信息
from langchain_core.messages import RemoveMessage

# Message list
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Isolate messages to delete
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)
add_messages(messages , delete_messages)
```



## 多模式

  现在，让我们在图中使用特定的 `input` 和 `output` 模式。

这里，`input` / `output` 模式对图的输入和输出中允许的键进行**过滤**。

此外，我们可以使用类型提示 `state: InputState` 来指定每个节点的输入模式。

当图使用多个模式时，这一点尤为重要。

例如，我们使用以下类型提示来表明 `answer_node` 的输出将被过滤为 `OutputState`。

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"question":"hi"})
```

## 过滤和修剪消息

###  消息过滤

通过 `RemoveMessage` 删除消息

```python
from langchain_core.messages import RemoveMessage

# Nodes
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    # 删除除最近两条消息之外的所有消息。
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

查看效果：

```python
# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Invoke
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```



如果不需要或不想修改图状态，您可以直接过滤传递给聊天模型的消息。

例如，只需将过滤后的列表：`llm.invoke(messages[-1:])` 传递给模型即可。

```python
# Node
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

### 消息修剪

另一种方法是根据预设的词元数量[修剪消息](https://docs.langchain.com/oss/python/langgraph/add-memory#trim-messages)。

这会将消息历史记录限制在指定数量的词元内。

过滤仅返回代理之间消息的后验子集，而修剪则限制了聊天模型可用于响应的词元数量。

请参阅下面的 `trim_messages`。

```python
from langchain_core.messages import trim_messages

# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```



## 带有消息摘要和外部数据库内存的聊天机器人

### 导包

```python
import os, getpass # 和配置 key 相关
from langchain_ollama import ChatOllama # 模型相关
from langsmith import traceable # LangSmith 相关
from langgraph.graph import MessagesState # 追加到消息列表 以 MessagesState 作为状态来传递 可以点进去看看样子
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage # 用户信息和系统信息和删除信息
from langgraph.graph import START, END, StateGraph # 创建 图 相关
from langgraph.prebuilt import tools_condition # 如果 LLM 决定调用工具，通向叫 "tools" 的节点，否则去 END 这部分没用上
from langgraph.prebuilt import ToolNode # 内置的 ToolNode 组件，只需传入工具列表即可初始化它，相当于一个节点 这部分没用上
from IPython.display import Image, display # 展示相关
from langgraph.checkpoint.memory import MemorySaver # Agent memory 相关 这部分就是用的 SqliteSaver 来代替的
import sqlite3 # 使用小巧、快速、流行的数据库 SQLite
from langgraph.checkpoint.sqlite import SqliteSaver # 和 memory 相关
from typing_extensions import Literal # 必须是其中的字段
```

### 导入环境变量

```python
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# LangSmith 相关
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

### 使用 SQLite 来持久化 数据

```python
# In memory
# 在内存中创建一个数据库，关闭线程检测，共享这个数据库
# conn = sqlite3.connect(":memory:", check_same_thread = False)

# 如果提供路径，就会创建一个数据库
db_path = r"D:\code\langchain\langchain-academy\module-2\state_db\test.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
```

定义 memory 和数据库绑定

```python
memory = SqliteSaver(conn)
```

### 定义聊天机器人

```python
qwen = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

# 自定义状态，加上个总结字段
class State(MessagesState):
    summary: str

# 定义调用模型的逻辑
def call_model(state: State):
    # 如果存在摘要，获得摘要
    summary = state.get("summary","")

    # 如果有摘要，加入进去
    if summary:

        # 将摘要添加进系统提示信息
        system_message = f"先前对话的总结：{summary}"

        # 将摘要添加到任何较新的消息中(放在最前面，最先读到系统提示)
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    response = qwen.invoke(messages)
    return {"messages": response}

# 总结对话内容
def summarize_conversation(state: State):

    # 首先，要获得任何存在的摘要
    summary = state.get("summary","")

    # 创建自己的摘要模板
    if   :

        # 已经存在摘要
        summary_message = (
            f"这是迄今为止的对话摘要：{summary}\n\n"
            "请根据以上新消息补充摘要："
        )

    else:
        summary_message = "请总结以上对话内容："

    # 在历史记录中添加提示
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = qwen.invoke(messages)

    # 删除除最近两条消息之外的所有消息
    # 把前面的总结了就不需要留下来了
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    # 把摘要存起来，然后删除多余的消息
    return {"summary": response.content, "messages": delete_messages}

# 决定是结束对话还是总结对话
def should_continue(state: State) -> Literal ["summarize_conversation", END]:
    """
    返回要执行的下一个节点
    """
    messages = state["messages"]

    # 如果超过六条信息，就对对话进行总结
    if len(messages) > 6:
        return "summarize_conversation"

    return END
```



### 使用 SQLite Checkpointer 来构建图

```python
# 定义 一个 新的 图
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# 设置入口点为 conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("conversation", END)

# 组合
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```



### 测试效果

创建一个线程并多次调用

```python
# 创建一个线程
config = {"configurable": {"thread_id": "1"}}

# 开始对话
input_message = HumanMessage(content="你好，我是文轶")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="你还记得我的名字吗？")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="我喜欢玩博德之门3！")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

查看所有聊天记录

```python
for m in output['messages']:
    m.pretty_print()
```

确认一下状态是否已经在本地保存

可以重启内核后再次调用试试

```python
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
graph_state
```

重启后打印全部历史记录

```python
# 1. 获取 snapshot 对象 (包裹)
graph_state = graph.get_state(config)

# 2. 打开包裹，拿出 values 字典 (里面的东西)
all_values = graph_state.values 
# 此时 all_values 类似： {'messages': [HumanMessage(...), AIMessage(...)]}

# 3. 从字典里取出 "messages" 列表
chat_history = all_values["messages"]

# --- 打印出来看看 ---
for msg in chat_history:
    msg.pretty_print() # LangChain 自带的漂亮打印方法
```



# Module-3

## 生成分析师：人机协作

### 创建分析师并使用人机协作对其进行审核

```python
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Analyst(BaseModel):
    affiliation: str = Field(
        description="分析师的主要所属机构。",
    )
    name: str = Field(
        description="分析师的名字。"
    )
    role: str = Field(
        description="分析师在该主题中的角色。",
    )
    description: str = Field(
        description="对分析师的关注点和行为的描述。",
    )
    # @property 装饰器将一个方法伪装成“属性”。不需要像调用函数那样写 analyst.persona()，而是直接写 analyst.persona。
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel): # 提前设置好的 结构化输出
    analysts: List[Analyst] = Field( # Field 添加额外的约束、默认值或描述信息
        description="分析师完整名单，包括他们的职务和所属机构。",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
```

### 构建图结构并且设置断点

```python
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import Literal

analyst_instructions="""你的任务是创建一组人工智能分析师角色。请仔细遵循以下说明:

1. 首先，回顾研究课题:
{topic}
        
2. 查阅所有可选的、用于指导分析报告撰写的编辑反馈意见: 
        
{human_analyst_feedback}
    
3. 根据以上文件和/或反馈，确定最有趣的主题。
                    
4. 选择排名前 {max_analysts} 的主题。

5. 为每个主题指派一名分析师。""" 

def create_analysts(state: GenerateAnalystsState):
    
    """ 创建分析师 """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
        
    # 强制结构化输出
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts)

    # Generate question 
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="生成分析师名单。")])
    
    # Write the list of analysis to state  
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ 应该中断的空操作节点 """
    pass

def should_continue(state: GenerateAnalystsState) -> Literal ["create_analysts", END]:
    """ 返回要执行的下一个节点 """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    
    # Otherwise end
    return END

# Add nodes and edges 
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

# Compile
memory = MemorySaver()
# interrupt_before: 设置断点
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# View
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

### 定义输入并且 利用 thread 来回溯和保存记忆

```python
# Input
max_analysts = 3 
topic = "采用 LangGraph 作为代理框架的优势"
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
# 流式输出配合 interrupt_before 设置断点
for event in graph.stream({"topic":topic,"max_analysts":max_analysts,}, thread, stream_mode="values"):
    # Review
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)  
```

查看下一个将要运行的节点

```python
# Get state and look at next node
state = graph.get_state(thread)
state.next
```

更新 信息

```python
# We now update the state as if we are the human_feedback node
# as_node: 这次状态更新，请视作是由 human_feedback 这个节点完成的。
# 1、触发条件边：之前定义的 add_conditional_edges("human_feedback", ...) 是以该节点结束为触发点的。

# 2、历史追溯：在图形化的轨迹（Trace）中，你会看到状态的改变归功于这个节点，逻辑链路非常清晰。
graph.update_state(thread, {"human_analyst_feedback": 
                            "再加入一位来自创业公司的人员，以增加创业者的视角。"}, as_node="human_feedback")
```

继续执行

```python
# Continue the graph execution
# 输入设置为 None 就可以接着上次断掉的地方重连
for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50) 
```



```python
# If we are satisfied, then we simply supply no feedback
# 如果满意，不反馈就行
further_feedack = None
graph.update_state(thread, {"human_analyst_feedback": 
                            further_feedack}, as_node="human_feedback")
```

```python
# Continue the graph execution to end
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)
```

查看最后的状态和 分析师列表

```python
final_state = graph.get_state(thread)
analysts = final_state.values.get('analysts')
```

查看 下一个节点

```python
final_state.next # 应该是 ()  代表到 END 了
```

查看最后生成的分析师

```python
for analyst in analysts:
    print(f"Name: {analyst.name}")
    print(f"Affiliation: {analyst.affiliation}")
    print(f"Role: {analyst.role}")
    print(f"Description: {analyst.description}")
    print("-" * 50) 
```



## 注释

### `stream_mode="values"` (全量状态模式)

该模式返回的是**当前节点执行完后，整个 State（状态）的完整快照**。

- **返回内容**：一个包含当前状态下**所有字段**的字典。
- **特点**：让你看到此时此刻图的“全貌”。无论当前节点改动了多少，它都会把没改动的字段也一起发给你。
- **适用场景**：记录日志、在 UI 界面上实时同步显示完整的表单或数据状态、需要获取全局上下文的情况。



### `stream_mode="updates"` (增量更新模式)

这是你代码中使用的模式，它只返回**当前节点执行后所产生的改变**。

- **返回内容**：一个字典，键是**节点名称**，值是该节点**返回的特定字段**。
- **特点**：非常清晰地展示了“谁做了什么”。如果一个节点只更新了 `analysts` 字段，你就只会收到 `analysts` 的数据，而不会收到 `topic` 等其他未变动的数据。
- **适用场景**：调试逻辑、监控特定节点的产出、或者在前端只针对变化部分做动画提示



# 集成 MCP

```bash
pip install "langgraph-api>=0.2.3" "langgraph-sdk>=0.1.61"
pip install langchain-mcp-adapters
```

```python
from langgraph.prebuilt import create_react_agent # 推荐使用 LangGraph 构建 Agent
```

## 以下半成品，还得接着改！！！

```python
# 高德的 MCP 服务端 配置
gaode_mcp_server_config = {
    "url":"https://mcp.api-inference.modelscope.net/29a554731f6b49/mcp",
    "transport":"streamable_http",
}
from langchain_mcp_adapters.client import MultiServerMCPClient

# 创建客户端
mcp_client = MultiServerMCPClient(
    {
        "gaode_mcp_server_config": gaode_mcp_server_config,
    }
)

from langchain_openai import ChatOpenAI

# 使用 ChatOpenAI 类，但是指向 DeepSeek 的服务器
llm = ChatOpenAI(
    model="qwen-plus",  # 或者 "deepseek-coder"
    api_key="sk-855eeb2da26c4a0dafb9e34990ea3d10",  # 你的 DeepSeek API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 关键点：修改 Base URL
    temperature=0
)

import asyncio
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

# 1. 定义状态
class State(TypedDict):
    # 消息列表，add_messages 表示追加模式
    messages: Annotated[list, add_messages]

async def create_agent_graph(mcp_client, llm):
    # 获取工具
    tools = await mcp_client.get_tools()
    # 2. 定义节点逻辑
    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State):
        # 这里可以插入 System Prompt
        system_message = "你是一个使用高德地图服务的智能助手。请根据用户的请求，调用相应的工具来获取信息。"
        return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

    # 3. 构建图
    graph_builder = StateGraph(State)
    
    # 添加节点
    graph_builder.add_node("agent", chatbot)
    tool_node = ToolNode(tools) # LangGraph 自带的工具执行节点
    graph_builder.add_node("tools", tool_node)
    
    # 添加边
    graph_builder.add_edge(START, "agent")
    
    # 条件边：决定是继续对话还是结束
    def should_continue(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        # 如果 LLM 返回了 tool_calls，就去 tools 节点
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    graph_builder.add_conditional_edges("agent", should_continue)
    graph_builder.add_edge("tools", "agent") # 工具执行完回 agent

    # 4. 编译 (在这里注入记忆！)
    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)
    
    return app

# 使用
agent = asyncio.run(create_agent_graph(mcp_client, llm))

```

