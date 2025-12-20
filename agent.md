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



# Agent 综述

## 2 智能体方法论

### 2.1 智能体构建

#### 2.1.3 规划能力

第 2.1.3 部分展示了 LLM 智能体如何从简单的线性执行者进化为复杂的规划者。通过**任务分解**（从静态链到动态树）和**反馈循环**（利用环境、人类和多智能体输入），智能体能够处理更长、更复杂且容错率更低的任务。

##### 2.1.3.1 任务分解策略

这是增强 LLM 规划能力的基本方法，即将复杂问题拆解为更易于管理的子任务 。虽然解决整个大问题可能很困难，但智能体可以更容易地处理子任务，并整合结果 。

论文将分解策略分为两类：**单路径链式 (Single-path chaining)** 和 **多路径树状扩展 (Multi-path tree expansion)** 。

###### A 单路径链式

- **基本形式：** 最简单的版本是零思维样本

- **”计划-求解“范式：**首先要求智能体制定一个由一系列相互构建的子任务组成的计划，然后按顺序解决这些子任务 。这种方法简单易行 

- **局限性**：缺乏灵活性，且容易在链条中积累错误，因为智能体必须严格遵循预定义计划 。

- **改进方案：**

  - **动态规划：**采用仅基于当前情况生成下一个子任务的动态规划（如 **ReAct**）。这使智能体能够接收环境反馈并相应调整计划，增强鲁棒性和适应性 。

  - **多思维链集成：**类似于集成学习方法，利用自我一致性（Self-consistency）、多数投票（Majority Voting）和智能体讨论（Agent Discussion）来结合多条思维链 。通过结合多条链的智慧，智能体可以做出更准确的决策并降低错误积累风险 。

###### B 多路径树状

- **结构**：使用树而不是链作为数据结构。在规划时存在多条可能的推理路径，允许智能体根据反馈信息进行回溯 。
- **代表方法：**
  - **思维树 (Tree of Thoughts, ToT)**：通过树状思维过程探索解空间，允许模型回溯到之前的状态并纠正错误，适用于涉及“试错-修正”过程的复杂任务 。
  - **蒙特卡洛树搜索 (MCTS)**：在更现实的场景中，利用强化学习和 MCTS 等高级算法，结合环境或人类反馈动态调整推理路径 。这有助于机器人在机器人技术和游戏等领域的应用 。

##### 2.1.3.2 反馈驱动的迭代

这是规划能力的另一个关键方面，使智能体能够从反馈中学习并随着时间的推移提高性能 。反馈机制形成了一个闭环，指导智能体更新计划、调整推理路径甚至修改目标，直到达成满意的计划 。

论文总结了四种主要的反馈来源：

- **环境反馈 (Environmental Feedback)**：
  - **来源**：具身智能体（Embodied Agent）操作的环境 。
  - **案例**：BrainBody-LLM 。
- **人类反馈 (Human Feedback)**：
  - **来源**：用户交互或预先准备的手动标记数据 。
  - **案例**：TrainerAgent 。
- **模型内省 (Model Introspection)**：
  - **来源**：由智能体自身生成，进行自我评估 。
  - **案例**：RASC 。
- **多智能体协作 (Multi-agent Collaboration)**：
  - **来源**：多个智能体协同解决问题并交换见解 。
  - **案例**：REVECA 

#### 2.1.4 动作执行

**如果智能体无法有效地执行计划，那么再好的计划也是无用的** 。动作执行赋予了 LLM 智能体在现实世界或虚拟环境中产生实际影响的能力。

论文将动作执行细分为两个主要维度：**工具利用 (Tool Utilization)** 和 **物理交互 (Physical Interaction)**。

##### 2.1.4.1 工具利用

这是解决 LLM 自身局限性（如无法进行精确计算、无法获取实时信息、代码生成能力受限等）的主要手段 。工具利用能力被进一步细分为两个关键决策过程：

- **工具使用决策 (Tool-use Decision)**：
  - **定义**：这是“是否”使用工具的判断过程 。
  - **触发机制**：当智能体在生成内容时信心不足，或者遇到特定功能性难题（如需要精确数学计算）时，应判定为需要调用工具 。
  - **代表工作**：TRICE , GPT4Tools 。
- **工具选择 (Tool Selection)**：
  - **定义**：这是“使用哪个”工具的判断过程 。
  - **核心挑战**：智能体需要理解工具的功能描述，并结合当前的上下文情境做出选择。
  - **优化策略**：例如 Yuan 等人提出的简化工具文档的方法，帮助模型更准确地理解可用工具，从而提高选择的准确性 。
  - **代表工作**：EASYTOOL , AvaTaR 。

##### 2.1.4.2 物理交互

这是**具身智能体 (Embodied LLM Agents)** 的基础，标志着智能体从纯数字空间走向物理世界 。

- **核心能力**：智能体不仅要输出文本或代码，还需要在现实世界中执行具体动作，并解释环境反馈 。
- **复杂性因素**：在现实世界部署时，智能体必须理解和处理多种复杂因素，包括：
  - **机器人硬件 (Robotic Hardware)**：硬件的物理限制和操作接口。
  - **社会知识 (Social Knowledge)**：在人类社会环境中行动所需的常识和规范 。
  - **多智能体交互**：与其他 LLM 智能体的协作或互动 。
- **代表工作**：DriVLMe  (结合具身和社会经验), ReAd , Collaborative Voyager 。

### 2.2 智能体协作

通过多智能体交互，可以利用**分布式智能**、协调行动并优化决策，从而超越单个智能体的推理局限 。

论文将协作范式归纳为三种基础架构：**中心化控制**、**去中心化协作**和**混合架构**。以下是详细分析：

#### 2.2.1 中心化控制

这种架构类似于公司的层级管理制度。

- **核心机制**：存在一个中央控制器（Central Controller），负责组织智能体活动、分配任务和整合决策 。子智能体通常只能与控制器通信，而不能相互直接通信 。
- **两种实现策略**：
  - **显式控制器系统 (Explicit Controller Systems)**：利用专门的协调模块（通常是独立的 LLM 智能体）来分解任务。
    - *案例*：**Coscientist** 由人类操作员作为中央控制器来管理科学实验流程 ；**MetaGPT** 分配专门的经理角色来控制软件开发的不同职能阶段 。
  - **基于分化的系统 (Differentiation-based Systems)**：通过提示词（Prompt）引导一个元智能体（Meta Agent）扮演不同的子角色。
    - *案例*：**AutoAct** 将元智能体分化为计划、工具和反思三个子智能体 ；**Meta-Prompting** 由单个模型充当协调者，动态分配子任务 。
- **适用场景**：适合需要严格协调的任务关键型场景，如工业自动化和科学研究 。

#### 2.2.2 去中心化协作

这种架构消除了中央节点，避免了单点瓶颈，类似于扁平化的团队或社交网络。

- **核心机制**：智能体之间通过自组织协议进行点对点的直接交互 。
- **两种主要方法**：
  - **基于修订的系统 (Revision-based Systems)**：
    - *特点*：智能体观察同伴生成的最终决策，并通过结构化的编辑协议迭代完善共享的输出 。这通常能产生更标准化的结果 。
    - *案例*：**MedAgents** 通过专家投票达成共识 ；**ReConcile** 通过相互分析和置信度评估来完善答案 。
  - **基于沟通的系统 (Communication-based Systems)**：
    - *特点*：结构更灵活，允许智能体直接对话并观察同伴的推理过程 。非常适合模拟人类社交互动等动态场景 。
    - *案例*：**AutoGen** 支持群聊模式进行迭代辩论 ；**MAD** 通过结构化协议解决“思维退化”问题 。

#### 2.2.3 混合架构

这种架构试图在“可控性”与“灵活性”之间取得平衡，优化资源利用率。

- **核心机制**：战略性地结合中心化协调和去中心化协作 。
- **两种实现模式**：
  - **静态系统 (Static Systems)**：预定义了固定的协作模式。
    - *案例*：**CAMEL** 在组内采用去中心化角色扮演，但在组间保持中心化治理 ；**AFlow** 采用了“中心化战略规划 + 去中心化战术谈判”的三层层级结构 。
  - **动态系统 (Dynamic Systems)**：引入神经拓扑优化器，根据实时反馈动态重构协作结构 。
    - *案例*：**DyLAN** 根据“智能体重要性评分”动态调整协作结构，以优化任务完成度 ；**MDAgents** 根据任务的复杂性（低、中、高）动态分配协作结构 。
