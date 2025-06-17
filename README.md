# DocuForge

基于 AI 的 PRD（产品需求文档）智能重写引擎，采用三阶段重写算法。

## 功能特性

- **结构优先**：创建带有空内容区块的初始文档大纲
- **顺序生成**：使用滚动上下文方法填充章节内容
- **闭环修正**：AI 驱动的审查和有针对性的修复
- **灵活输出**：支持 Markdown 和结构化 JSON 输出
- **进度跟踪**：重写过程中的实时反馈

## 安装

```bash
pip install docuforge
```

## 环境配置

创建 `.env` 文件并配置你的 Azure OpenAI：

```
AZURE_OPENAI_ENDPOINT=你的端点地址
AZURE_OPENAI_DEPLOYMENT_NAME=你的部署名称
AZURE_OPENAI_API_VERSION=你的API版本
```

## 使用方法

### 基本用法

```bash
# 重写文档并输出到标准输出
docuforge --original-doc document.md --clarifications clarify.json

# 保存输出到指定文件
docuforge --original-doc doc.md --clarifications clarify.json \
          --output-md rewritten.md --output-json structure.json

# 静默模式（无进度输出）
docuforge --original-doc doc.md --clarifications clarify.json --quiet
```

### 命令选项

**必需参数：**
- `--original-doc`：待重写的原始文档文件路径
- `--clarifications`：包含澄清问答对的 JSON 文件路径

**可选参数：**
- `--output-md`：保存重写后 Markdown 文档的路径
- `--output-json`：保存结构化文档 JSON 的路径
- `--max-revision-rounds`：最大修订轮数（默认：3）
- `--quiet`：抑制进度输出
- `--version`：显示版本信息

### 澄清文件格式

澄清文件应为包含问答对的 JSON 格式：

```json
[
  {
    "question": "这个功能的主要目标是什么？",
    "answer": "提升用户体验并增加参与度"
  },
  {
    "question": "目标用户群体是谁？",
    "answer": "企业客户和高级用户"
  },
  {
    "question": "关键成功指标是什么？",
    "answer": "用户采用率和任务完成时间"
  }
]
```

带包装对象的替代格式：

```json
{
  "clarifications": [
    {
      "question": "这解决了什么问题？",
      "answer": "用户在复杂工作流程中遇到困难"
    }
  ]
}
```

## 系统架构

系统实现三阶段重写算法：

1. **大纲和结构初始化** - 创建带有空内容区块的 `DocumentStructure`
2. **顺序内容填充** - 使用滚动上下文方法填充章节
3. **审查和修订** - AI 驱动的审查和有针对性的修复

核心组件：
- `RewriteChain` - 使用 LangGraph 的主要编排器
- `ContextBuilder` - 聚合原始内容和澄清信息
- `OutlineGenerator` - 创建初始文档结构
- `ContentFiller` - 带有上下文连续性的顺序章节生成
- `Reviser` - AI 审查和有针对性的修订系统

## 开发

### 测试 Azure OpenAI 集成

```bash
python example.py
```

### 技术栈

- **Python 3.10+** 配合 LangChain 生态系统
- **LangChain 0.3.25** - LLM 框架和提示管理
- **LangGraph 0.4.8** - 带条件边的有状态代理工作流
- **LangSmith 0.3.45** - 调试和可观测性
- **Pydantic 2.11.5** - 数据验证和结构化输出
- **Azure OpenAI** - GPT-4/O3 集成，推理努力程度设为"高"

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。