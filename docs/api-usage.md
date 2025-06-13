# API 使用说明

DocuForge 是一个AI驱动的PRD文档重写引擎，提供简洁的Python API供其他项目集成使用。

## 安装

```bash
pip install docuforge
```

## 基本用法

### 1. 简单调用

```python
from docuforge import RewriteChain, RewriteRequest, ClarificationItem

# 创建重写请求
request = RewriteRequest(
    original_content="原始PRD文档内容...",
    clarifications=[
        ClarificationItem(
            question="产品的核心功能是什么？",
            answer="用户管理和数据分析"
        )
    ]
)

# 创建重写链并执行
chain = RewriteChain()
result = chain.invoke(request)

# 获取结果
rewritten_content = result.rewritten_content  # 重写后的文档
structured_data = result.structured_document  # 结构化数据
```

### 2. 带进度回调的调用

```python
from docuforge import RewriteChain, DefaultCallbackHandler

# 使用默认回调处理器
callback = DefaultCallbackHandler()
chain = RewriteChain()
result = chain.invoke(request, callback_handler=callback)
```

### 3. 自定义回调处理器

```python
from docuforge import ProgressCallbackHandler

class MyCallbackHandler:
    def on_stage_start(self, stage_name: str, **kwargs):
        print(f"开始阶段: {stage_name}")
    
    def on_stage_end(self, stage_name: str, **kwargs):
        print(f"完成阶段: {stage_name}")
    
    def on_stage_progress(self, stage_name: str, message: str, **kwargs):
        print(f"[{stage_name}] {message}")

callback = MyCallbackHandler()
result = chain.invoke(request, callback_handler=callback)
```

### 4. 使用工厂函数创建

```python
from docuforge import create_rewrite_chain

# 使用Azure OpenAI配置
chain = create_rewrite_chain(
    azure_endpoint="your-azure-endpoint",
    azure_deployment="your-deployment-name", 
    api_version="2024-02-01"
)

result = chain.invoke(request)
```

## 数据结构

### 输入数据

```python
# 澄清问答项
ClarificationItem(
    question="问题内容",
    answer="答案内容"
)

# 重写请求
RewriteRequest(
    original_content="原始文档内容",
    clarifications=[...]  # ClarificationItem列表
)
```

### 输出数据

```python
# 重写结果
RewriteResult(
    rewritten_content="重写后的markdown文档",
    structured_document=DocumentStructure(
        title="文档标题",
        sections=[
            DocumentSection(
                title="章节标题",
                content="章节内容",
                level=1,
                order=0,
                goal="章节写作目标"
            )
        ],
        metadata={}
    )
)
```

## 环境配置

创建 `.env` 文件配置Azure OpenAI：

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-01
```

## 完整示例

```python
import os
from dotenv import load_dotenv
from docuforge import (
    RewriteChain, 
    RewriteRequest, 
    ClarificationItem,
    DefaultCallbackHandler
)

# 加载环境变量
load_dotenv()

# 准备数据
original_content = """
# 用户管理系统PRD
## 概述
需要开发一个用户管理系统...
"""

clarifications = [
    ClarificationItem(
        question="支持哪些用户角色？",
        answer="管理员、普通用户、访客三种角色"
    ),
    ClarificationItem(
        question="需要哪些核心功能？",
        answer="用户注册、登录、权限管理、数据统计"
    )
]

# 创建请求
request = RewriteRequest(
    original_content=original_content,
    clarifications=clarifications
)

# 执行重写
chain = RewriteChain()
callback = DefaultCallbackHandler()
result = chain.invoke(request, callback_handler=callback)

# 使用结果
print("重写完成！")
print(f"文档标题: {result.structured_document.title}")
print(f"章节数量: {len(result.structured_document.sections)}")

# 保存结果
with open("rewritten_document.md", "w", encoding="utf-8") as f:
    f.write(result.rewritten_content)
```

## 异常处理

```python
from docuforge.exceptions import RewriteError

try:
    result = chain.invoke(request)
except RewriteError as e:
    print(f"重写失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```