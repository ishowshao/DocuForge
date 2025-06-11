# 内容重写引擎技术方案

## 1. 系统概述

### 1.1 设计目标

内容重写引擎（DocuForge）是PRD Writer系统的核心组件，负责基于澄清信息对原始PRD文档进行智能重写和优化。设计目标包括：

- **功能纯粹**：专注于内容重写的核心功能，接口简洁清晰
- **内容优化**：基于澄清信息完善和优化文档内容表述  
- **结构清晰**：生成逻辑严谨、结构清晰的重写文档
- **灵活输出**：支持纯文本和结构化数据的输出格式

### 1.2 系统定位

内容重写引擎在PRD Writer系统中的定位：

**输入来源**：
- 原始文档内容（纯文本）
- 交互澄清获得的问答内容

**输出产物**：
- 重写后的高质量文档（结构化数据与纯文本）

**系统依赖**：
- AI基础设施（大语言模型调用）

## 2. 核心算法：三阶段重写

内容重写引擎的核心算法将摒弃简单的"全文重写"模式，转而采用一套更精细、更可控的、模拟人类专家写作流程的三阶段方法："**结构先行，顺序生成，闭环修正**"。

该算法将复杂的文档重写任务分解为三个独立的、连续的阶段，旨在最大化地利用大语言模型（LLM）在结构规划和上下文理解上的优势，同时通过自我修正机制，确保最终输出文档的质量、一致性和可控性。流程的核心是**创建并逐步完善一个`DocumentStructure`对象**。

### 2.1 阶段一：大纲与结构初始化 (Outline & Structure Initialization)

这是重写流程的"规划"阶段。

1.  **输入**：将原始文档内容和所有的澄清问答（Q&A）聚合在一起，作为统一的背景知识。
2.  **过程**：调用大语言模型（LLM），并指示其扮演"解决方案架构师"或"资深产品经理"的角色。模型的任务不是直接写作，而是通读全部背景知识，然后规划出一份新文档的**结构化大纲**。
3.  **输出**：产出的是一个初始化的 `DocumentStructure` 对象。该对象包含了文档标题和一份 `DocumentSection` 列表，其中每个 `section` 都已填充了标题（`title`）和该章节的核心写作目标（`goal`），但内容（`content`）为空。这一步确保了文档的"骨架"是正确且符合最终需求的。

### 2.2 阶段二：逐段内容填充 (Sequential Content Filling)

这是重写流程的"执行"阶段，其核心是填充上一阶段创建的结构。

1.  **输入**：上一阶段生成的、包含空内容的 `DocumentStructure` 对象，以及完整的原始背景知识。
2.  **过程**：引擎将严格按照 `DocumentStructure` 中 `sections` 的顺序，自上而下地、**逐一**为其填充内容。在生成当前章节（例如，第 N 章节）时，它会向LLM提供一个精心构造的上下文，该上下文包含：
    *   完整的原始背景知识（原始文档+Q&A）。
    *   前面 N-1 个已经生成好的章节的全部内容。
    *   第 N 章节的写作目标（来自大纲）。
3.  **输出**：一个所有章节 `content` 都已填充完毕的 `DocumentStructure` 对象。这份对象代表了完整的文档初稿。这种"上下文滚动"的生成方式，能最大限度地保证章节间的逻辑连贯性、术语一致性和行文风格统一。

### 2.3 阶段三：审核与修订 (Review and Revision)

这是重写流程的"质量保证"阶段，一个自我校对和修正的闭环。

1.  **输入**：上一阶段生成的、内容完整的 `DocumentStructure` (文档初稿)，以及原始的背景知识。
2.  **过程**：
    *   **评审（Review）**：首先，调用LLM扮演"文档评审专家"的角色。模型需要将初稿（通过序列化 `DocumentStructure` 得到）与原始背景知识进行比对，其目标是**找出问题**，例如：逻辑矛盾、信息遗漏、需求点未完全澄清、前后不一致等。评审结果应以结构化的形式（如问题列表）返回。
    *   **修订（Patch）**：如果评审发现了问题，引擎会启动"靶向修复"流程。它会遍历每一个被发现的问题，定位到 `DocumentStructure` 中具体的 `DocumentSection`，并再次调用LLM，下达一个非常具体的"修复指令"，要求其**仅重写该 `section` 的 `content`** 以修正该特定问题。
3.  **输出**：一份经过内部多轮审核和修订的、高质量的最终 `DocumentStructure` 对象。这个阶段确保了即使在生成过程中出现偏差，系统也有能力自我发现并纠正，从而极大地提升了最终结果的可靠性。

## 3. 架构与实现

为实现我们设计的"三阶段"智能重写算法，并将其构建为具备自主规划和修正能力的AI Agent，推荐采用业界主流的 `LangChain` 及 `LangGraph` 框架。

### 3.1 整体架构

整体架构直接映射"三阶段核心算法"的工作流，其组件清晰地反映了算法的每一个核心步骤。我们可以将整个 `RewriteChain` 映射为一个 `LangGraph` 计算图。

**核心组件**:
- **统一上下文构造器 (ContextBuilder)**：接收 `RewriteRequest`，将原始文档和澄清信息整合成统一的"背景知识"文本。
- **大纲生成器 (OutlineGenerator)**：对应算法第一阶段，基于背景知识调用AI模型生成初始化的`DocumentStructure`。
- **内容填充器 (ContentFiller)**：对应算法第二阶段，遵循大纲以"滚动上下文"模式为`DocumentStructure`填充各章节内容。
- **审核修订器 (Reviser)**：对应算法第三阶段，对初稿进行AI评审、识别问题并进行靶向修复。

### 3.2 基于LangGraph的实现

1.  **定义状态（State）**: 创建一个全局状态对象，在图的节点之间流转和更新。该状态包含所有关键数据，如：`original_content`、`clarifications`、`document_structure` (替代旧的`outline`和`filled_sections`) 和 `revision_issues` 等。
2.  **定义节点（Nodes）**: 上述核心组件将成为图中的节点，各自负责更新图的状态。
3.  **定义边（Edges）**:
    *   **常规边**: 连接 `OutlineGenerator` -> `ContentFiller` -> `Reviser`，形成基本工作流。
    *   **条件边（Conditional Edge）**: 实现自我修正的关键。在 `Reviser` 节点后进行条件判断：若 `revision_issues` 不为空，则将流程导向修复节点；否则，结束任务。

通过这种方式，`RewriteChain` 成为一个能够根据中间结果动态决定下一步行动的智能代理，实现了我们设计的核心算法，并增强了系统的健壮性和可观测性。

### 3.3 Prompt 设计要点

算法的成功高度依赖于Prompt的设计。

-   **大纲生成Prompt**：指示LLM扮演"解决方案架构师"，输出结构化的JSON，该JSON直接对应`DocumentStructure`的初始化结构。
-   **内容填充Prompt**：包含三部分关键信息：① 原始参考材料；② 已完成的上文章节；③ 当前章节的目标，确保内容连贯。
-   **审核修订Prompt**：指示LLM扮演"文档评审专家"，从多维度评审并以结构化格式报告问题，以便程序解析和执行`patch`。

### 3.4 健壮性与错误处理

为了保证引擎在真实环境中的稳定运行，必须考虑周全的错误处理机制。

- **API调用失败**：与LLM的所有交互都应包含重试逻辑（例如使用指数退避策略），以应对网络波动或临时的API服务中断。若多次重试后仍失败，应抛出明确的异常。
- **输出格式验证**：每个要求LLM输出结构化数据（如JSON）的步骤，都必须有严格的验证和解析环节。如果返回的数据格式不正确，应启动一个"格式修复"子流程：将错误信息和原始请求一并返回给LLM，要求其修正格式后重新输出。如果多次修正无效，则判定该阶段失败。
- **修正循环控制**：阶段三的"审核-修订"循环必须设置一个最大迭代次数上限（例如3次）。这可以防止因模型固执己见或问题无法修复而导致的无限循环，确保任务最终能够终止。
- **状态报告**：在发生无法恢复的错误时，引擎应清晰地报告失败的阶段和原因，这对于上层调用者和问题排查至关重要。

### 3.5 技术栈选型

为支持我们设计的"三阶段"重写算法及Agentic架构，我们推荐以下经过审慎选择的技术栈：

-   **编程语言：Python 3.10+**
    -   **理由**：Python 是AI领域的通用语言，拥有无与伦比的生态系统。所有主流的LLM框架和库（如 `LangChain`）都以Python为中心。其成熟的异步支持（`asyncio`）也非常适合构建IO密集型的AI应用。

-   **核心AI框架：LangChain & LangGraph**
    -   **理由**：`LangChain` 提供了与LLM交互、Prompt管理和输出解析的标准化高级抽象，极大地简化了AI应用的开发。`LangGraph` 专为构建有状态、可循环的复杂代理而设计，完美契合我们"结构先行，闭环修正"的算法思想，并提供了强大的流程控制和可观测性。

-   **数据模型与验证：Pydantic V2**
    -   **理由**：在整个重写流程中，数据对象的结构正确性至关重要。Pydantic通过Python类型注解提供了业界领先的数据验证、解析和序列化能力。它能与`LangChain`的输出解析器无缝集成，确保从LLM获取的数据严格符合我们预定义的 `DocumentStructure` 等模型，是保障系统健壮性的基石。


## 4. 接口设计与数据结构

### 4.1 核心数据模型

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol

# --- 输入与输出 ---
@dataclass
class ClarificationItem:
    """单个澄清问答对"""
    question: str
    answer: str

@dataclass
class RewriteRequest:
    """重写请求数据结构"""
    original_content: str
    clarifications: List[ClarificationItem]

@dataclass
class RewriteResult:
    """重写结果数据结构"""
    rewritten_content: str  # 由下方的 structured_document 序列化生成
    structured_document: 'DocumentStructure'

# --- 内部结构化数据 ---
@dataclass
class DocumentStructure:
    """文档结构化数据"""
    title: str
    sections: List['DocumentSection']
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentSection:
    """文档章节"""
    title: str
    content: str
    level: int
    order: int
    goal: str  # 新增：章节的写作目标，在阶段一生成
```

### 4.2 进度反馈回调

为提供清晰的状态可见性，引擎通过回调机制报告进度。调用方可提供一个回调处理器，引擎在各阶段关键节点会调用其方法。

```python
class ProgressCallbackHandler(Protocol):
    """进度回调处理器的接口协议"""

    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        """在一个阶段开始时调用"""
        ...

    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        """在一个阶段结束时调用"""
        ...

    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """
        报告一个阶段内部的、更细粒度的进度。
        参数:
        - message (str): 人类可读的进度文本。例如："正在生成第 2/5 节..."。
        """
        ...
```

### 4.3 引擎调用接口

`RewriteChain`的调用接口将整合请求对象和可选的回调处理器。

```python
class RewriteChain:
    # ... 内部实现 ...
    def invoke(self, request: RewriteRequest, callback_handler: Optional[ProgressCallbackHandler] = None) -> RewriteResult:
        # ... 实现细节 ...
        # 示例：在各阶段调用回调
        if callback_handler:
            callback_handler.on_stage_start("outline_generation")
        # ... 生成大纲 ...
        if callback_handler:
            callback_handler.on_stage_end("outline_generation")
        
        return ...
```

## 5. 命令行工具 (CLI)

为了方便引擎的独立测试、调试和集成，我们设计一个配套的CLI工具。它封装了引擎核心逻辑，负责处理文件IO和参数解析。

### 5.1 命令设计

```bash
# 使用示例
python -m prd_writer.rewrite \
    --original-doc ./path/to/original_document.md \
    --clarifications ./path/to/clarifications.json \
    --output-md ./path/to/rewritten_document.md \
    --output-json ./path/to/structured_output.json
```

-   `--original-doc` (必需): 原始文档路径。
-   `--clarifications` (必需): 包含澄清问答的JSON文件路径。JSON文件应是一个包含 `question` 和 `answer` 字段对象的列表。
-   `--output-md` (可选): Markdown格式的输出文件路径。如果未提供，则不输出此文件。
-   `--output-json` (可选): 结构化JSON的输出文件路径。如果未提供，则不输出此文件。
-   如果`--output-md`和`--output-json`均未提供，则将Markdown内容输出到标准输出（`stdout`）。

### 5.2 实现思路

1.  **参数解析**: 使用Python内置的`argparse`库解析命令行参数。
2.  **文件读取**: 读取`--original-doc`和`--clarifications`文件内容。
3.  **引擎调用**: 
    -   根据读取内容，实例化`RewriteRequest`对象。
    -   实例化一个`CLICallbackHandler`（将进度打印到`stderr`）。
    -   调用`RewriteChain.invoke(request, callback_handler)`。
4.  **结果处理**: 根据命令行参数，将返回结果中的`rewritten_content`和`structured_document`分别写入指定文件或打印到`stdout`。

### 5.3 CLI进度反馈

CLI工具将实现`ProgressCallbackHandler`接口，把所有进度信息格式化后打印到标准错误流（`stderr`），确保进度日志和最终产出分离。

**CLI回调实现示例**:
```python
import sys
from typing import Any

class CLICallbackHandler:
    """用于CLI的默认回调处理器"""
    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        print(f"INFO: Stage start: {stage_name}...", file=sys.stderr)

    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        print(f"INFO: Stage end: {stage_name}.", file=sys.stderr)

    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        print(f"INFO: [{stage_name}] {message}", file=sys.stderr)
```

