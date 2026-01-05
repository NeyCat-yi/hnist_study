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
from langgraph.graph import MessagesState # 追加到消息列表 以 MessagesState 作为状态来传递
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

