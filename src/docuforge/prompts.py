"""Prompt engineering and management for DocuForge."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from .models import DocumentSection, DocumentStructure


class PromptType(str, Enum):
    """Types of prompts used in the system."""
    OUTLINE_GENERATION = "outline_generation"
    CONTENT_FILLING = "content_filling"
    DOCUMENT_REVIEW = "document_review"
    SECTION_REVISION = "section_revision"


class PromptConfig(BaseModel):
    """Configuration for a specific prompt."""
    
    name: str = Field(description="Name of the prompt")
    version: str = Field(description="Version of the prompt")
    prompt_type: PromptType = Field(description="Type of the prompt")
    template: str = Field(description="The prompt template")
    system_message: Optional[str] = Field(default=None, description="System message template")
    variables: List[str] = Field(default_factory=list, description="Required template variables")
    description: str = Field(description="Description of what this prompt does")


class BasePromptBuilder(ABC):
    """Base class for prompt builders."""
    
    @abstractmethod
    def build_system_prompt(self, **kwargs: Any) -> str:
        """Build system prompt."""
        pass
    
    @abstractmethod
    def build_human_prompt(self, **kwargs: Any) -> str:
        """Build human prompt."""
        pass
    
    @abstractmethod
    def get_required_variables(self) -> List[str]:
        """Get list of required variables for this prompt."""
        pass


class OutlinePromptBuilder(BasePromptBuilder):
    """Builds prompts for outline generation."""
    
    def __init__(self, language: str = "zh"):
        """Initialize outline prompt builder.
        
        Args:
            language: Language for prompts (zh/en)
        """
        self.language = language
    
    def build_system_prompt(self, **kwargs: Any) -> str:
        """Build system prompt for outline generation."""
        if self.language == "zh":
            return """你是一个资深的产品经理和解决方案架构师。你的任务是分析原始文档内容和澄清信息，然后设计出一个清晰、逻辑严谨的文档结构大纲。

核心原则：
1. 结构先行：先设计清晰的文档骨架，再考虑内容填充
2. 逻辑严谨：章节之间要有清晰的逻辑关系和递进性
3. 目标明确：每个章节都要有明确的写作目标和预期效果
4. 层次清楚：合理使用标题层级，避免过深或过浅

输出要求：
- 严格按照JSON格式输出DocumentStructure对象
- content字段必须为空字符串（后续阶段填充）
- goal字段要具体描述该章节的写作目标
- 章节order从0开始连续递增
- level控制在1-4之间，保持合理的层次结构

JSON格式示例：
```json
{
  "title": "文档标题",
  "sections": [
    {
      "title": "章节标题",
      "content": "",
      "level": 1,
      "order": 0,
      "goal": "具体的写作目标描述"
    }
  ],
  "metadata": {"generated_by": "outline_generator", "version": "1.0"}
}
```"""
        else:
            return """You are a senior product manager and solution architect. Your task is to analyze the original document content and clarification information, then design a clear and logically rigorous document structure outline.

Core Principles:
1. Structure First: Design a clear document skeleton before considering content filling
2. Logical Rigor: Clear logical relationships and progression between sections
3. Clear Objectives: Each section must have clear writing goals and expected outcomes
4. Clear Hierarchy: Reasonable use of heading levels, avoiding too deep or too shallow structure

Output Requirements:
- Strictly output DocumentStructure object in JSON format
- content field must be empty string (filled in subsequent stages)
- goal field should specifically describe the writing objective of the section
- Section order starts from 0 and increments consecutively
- level should be controlled between 1-4, maintaining reasonable hierarchy

JSON Format Example:
```json
{
  "title": "Document Title",
  "sections": [
    {
      "title": "Section Title",
      "content": "",
      "level": 1,
      "order": 0,
      "goal": "Specific writing objective description"
    }
  ],
  "metadata": {"generated_by": "outline_generator", "version": "1.0"}
}
```"""
    
    def build_human_prompt(self, context: str, **kwargs: Any) -> str:
        """Build human prompt for outline generation."""
        if self.language == "zh":
            return f"""请基于以下背景信息，设计一个清晰的文档结构大纲：

{context}

分析要求：
1. 仔细阅读原始文档内容，理解核心主题和目标
2. 结合澄清信息，明确具体需求和关注点
3. 设计合理的文档层次结构，确保逻辑清晰
4. 为每个章节定义明确的写作目标

请输出结构化的文档大纲（JSON格式）："""
        else:
            return f"""Please design a clear document structure outline based on the following background information:

{context}

Analysis Requirements:
1. Carefully read the original document content to understand the core theme and objectives
2. Combine clarification information to clarify specific requirements and focus points
3. Design a reasonable document hierarchy to ensure clear logic
4. Define clear writing objectives for each section

Please output a structured document outline (JSON format):"""
    
    def get_required_variables(self) -> List[str]:
        """Get required variables for outline generation."""
        return ["context"]


class ContentPromptBuilder(BasePromptBuilder):
    """Builds prompts for content filling."""
    
    def __init__(self, language: str = "zh"):
        """Initialize content prompt builder.
        
        Args:
            language: Language for prompts (zh/en)
        """
        self.language = language
    
    def build_system_prompt(self, **kwargs: Any) -> str:
        """Build system prompt for content filling."""
        if self.language == "zh":
            return """你是一个专业的技术文档撰写专家。你的任务是为文档的特定章节撰写高质量的内容。

写作原则：
1. 目标导向：严格按照章节目标进行写作，不偏离主题
2. 上下文连贯：充分考虑前文内容，保持逻辑连贯性
3. 内容充实：提供有价值的、具体的信息，避免空洞表述
4. 结构清晰：使用合适的段落组织和标记，便于阅读
5. 语言专业：使用准确、专业的术语和表达

写作要求：
- 只输出章节正文内容，不包含章节标题
- 基于原始需求和澄清信息进行写作
- 考虑前面章节的内容，保持整体一致性
- 内容要有足够的深度和广度
- 使用Markdown格式进行排版

避免的问题：
- 重复前面章节的内容
- 偏离章节写作目标
- 内容过于简略或表面化
- 逻辑不清晰或结构混乱"""
        else:
            return """You are a professional technical documentation expert. Your task is to write high-quality content for specific document sections.

Writing Principles:
1. Goal-Oriented: Write strictly according to section objectives, stay on topic
2. Contextual Coherence: Fully consider preceding content, maintain logical coherence
3. Rich Content: Provide valuable, specific information, avoid empty statements
4. Clear Structure: Use appropriate paragraph organization and markers for readability
5. Professional Language: Use accurate, professional terminology and expressions

Writing Requirements:
- Output only section body content, not including section titles
- Write based on original requirements and clarification information
- Consider content from previous sections to maintain overall consistency
- Content should have sufficient depth and breadth
- Use Markdown format for layout

Problems to Avoid:
- Repeating content from previous sections
- Deviating from section writing objectives
- Content that is too brief or superficial
- Unclear logic or chaotic structure"""
    
    def build_human_prompt(
        self,
        section: DocumentSection,
        original_context: str,
        previous_content: str = "",
        document_title: str = "",
        **kwargs: Any
    ) -> str:
        """Build human prompt for content filling."""
        if self.language == "zh":
            prompt_parts = [
                f"文档标题：{document_title}",
                "",
                f"当前章节：{section.title}",
                f"章节目标：{section.goal}",
                f"章节层级：{section.level}",
                f"章节顺序：{section.order}",
                "",
                "=== 原始背景信息 ===",
                original_context,
                ""
            ]
            
            if previous_content and previous_content.strip():
                prompt_parts.extend([
                    "=== 前面章节的内容（用于保持连贯性）===",
                    previous_content,
                    ""
                ])
            
            prompt_parts.extend([
                "写作指导：",
                f"- 围绕章节目标：{section.goal}",
                "- 结合原始背景信息中的相关要求",
                "- 保持与前文的逻辑连贯性",
                "- 提供有价值的、具体的内容",
                "",
                "请为当前章节撰写内容（只输出正文，不包含标题）："
            ])
        else:
            prompt_parts = [
                f"Document Title: {document_title}",
                "",
                f"Current Section: {section.title}",
                f"Section Objective: {section.goal}",
                f"Section Level: {section.level}",
                f"Section Order: {section.order}",
                "",
                "=== Original Background Information ===",
                original_context,
                ""
            ]
            
            if previous_content and previous_content.strip():
                prompt_parts.extend([
                    "=== Content from Previous Sections (for coherence) ===",
                    previous_content,
                    ""
                ])
            
            prompt_parts.extend([
                "Writing Guidelines:",
                f"- Focus on section objective: {section.goal}",
                "- Integrate relevant requirements from original background",
                "- Maintain logical coherence with preceding content",
                "- Provide valuable, specific content",
                "",
                "Please write content for the current section (body only, no title):"
            ])
        
        return "\n".join(prompt_parts)
    
    def get_required_variables(self) -> List[str]:
        """Get required variables for content filling."""
        return ["section", "original_context", "document_title"]


class ReviewPromptBuilder(BasePromptBuilder):
    """Builds prompts for document review."""
    
    def __init__(self, language: str = "zh"):
        """Initialize review prompt builder.
        
        Args:
            language: Language for prompts (zh/en)
        """
        self.language = language
    
    def build_system_prompt(self, **kwargs: Any) -> str:
        """Build system prompt for document review."""
        if self.language == "zh":
            return """你是一个专业的文档评审专家。你需要仔细审查文档内容，识别可能存在的问题并提出改进建议。

评审维度：
1. 逻辑一致性：检查内容是否前后一致，无矛盾之处
2. 信息完整性：确认是否遗漏了重要信息或需求点
3. 结构合理性：评估章节组织是否合理，层次是否清晰
4. 内容准确性：验证是否偏离了原始需求和澄清信息
5. 语言质量：检查表达是否专业、准确、清晰

问题类型定义：
- inconsistency: 逻辑不一致或矛盾
- missing_info: 缺失重要信息
- structural_issue: 结构组织问题
- accuracy_issue: 内容准确性问题
- quality_issue: 语言质量问题

输出格式：
必须严格按照以下JSON格式输出RevisionReport对象：
```json
{
  "issues": [
    {
      "section_order": 0,
      "issue_type": "inconsistency",
      "description": "具体问题描述",
      "suggestion": "详细的修改建议"
    }
  ],
  "overall_quality": "整体质量评价"
}
```

注意：如果没有发现问题，issues数组应为空列表[]。"""
        else:
            return """You are a professional document review expert. You need to carefully examine document content, identify potential issues, and provide improvement suggestions.

Review Dimensions:
1. Logical Consistency: Check if content is internally consistent without contradictions
2. Information Completeness: Confirm if important information or requirements are missing
3. Structural Reasonableness: Evaluate if section organization is reasonable and hierarchy is clear
4. Content Accuracy: Verify if it deviates from original requirements and clarifications
5. Language Quality: Check if expression is professional, accurate, and clear

Issue Type Definitions:
- inconsistency: Logical inconsistency or contradiction
- missing_info: Missing important information
- structural_issue: Structural organization problem
- accuracy_issue: Content accuracy problem
- quality_issue: Language quality issue

Output Format:
Must strictly output RevisionReport object in the following JSON format:
```json
{
  "issues": [
    {
      "section_order": 0,
      "issue_type": "inconsistency",
      "description": "Specific issue description",
      "suggestion": "Detailed modification suggestion"
    }
  ],
  "overall_quality": "Overall quality assessment"
}
```

Note: If no issues are found, the issues array should be an empty list []."""
    
    def build_human_prompt(
        self,
        document_content: str,
        original_context: str,
        **kwargs: Any
    ) -> str:
        """Build human prompt for document review."""
        if self.language == "zh":
            return f"""请评审以下文档内容，识别可能存在的问题：

=== 原始需求和背景 ===
{original_context}

=== 生成的文档内容 ===
{document_content}

评审任务：
1. 仔细对比原始需求和生成的文档
2. 检查文档的逻辑一致性和完整性
3. 识别任何偏离要求或存在问题的地方
4. 为每个问题提供具体的修改建议

请输出评审报告（JSON格式）："""
        else:
            return f"""Please review the following document content and identify potential issues:

=== Original Requirements and Background ===
{original_context}

=== Generated Document Content ===
{document_content}

Review Tasks:
1. Carefully compare original requirements with generated document
2. Check document logical consistency and completeness
3. Identify any deviations from requirements or existing problems
4. Provide specific modification suggestions for each issue

Please output review report (JSON format):"""
    
    def get_required_variables(self) -> List[str]:
        """Get required variables for document review."""
        return ["document_content", "original_context"]


class RevisionPromptBuilder(BasePromptBuilder):
    """Builds prompts for section revision."""
    
    def __init__(self, language: str = "zh"):
        """Initialize revision prompt builder.
        
        Args:
            language: Language for prompts (zh/en)
        """
        self.language = language
    
    def build_system_prompt(self, **kwargs: Any) -> str:
        """Build system prompt for section revision."""
        if self.language == "zh":
            return """你是一个专业的文档修订专家。你需要根据发现的具体问题，对文档章节进行针对性的修改。

修订原则：
1. 精准修改：只修改存在问题的部分，保持其他内容不变
2. 问题导向：严格按照识别出的问题进行修改
3. 保持一致性：确保修改后的内容与整体文档风格一致
4. 质量提升：修改后的内容应该比原内容更好
5. 逻辑清晰：修改要符合逻辑，不引入新的问题

修订要求：
- 仔细分析问题描述和修改建议
- 基于原始背景信息进行修订
- 只输出修改后的章节内容，不包含标题
- 保持原有的章节目标和定位
- 使用Markdown格式进行排版

注意事项：
- 不要偏离章节的基本目标
- 不要引入与问题无关的修改
- 确保修改后内容的完整性和可读性"""
        else:
            return """You are a professional document revision expert. You need to make targeted modifications to document sections based on identified specific issues.

Revision Principles:
1. Precise Modification: Only modify problematic parts, keep other content unchanged
2. Problem-Oriented: Strictly modify according to identified issues
3. Maintain Consistency: Ensure revised content is consistent with overall document style
4. Quality Improvement: Revised content should be better than original content
5. Clear Logic: Modifications should be logical and not introduce new problems

Revision Requirements:
- Carefully analyze issue descriptions and modification suggestions
- Revise based on original background information
- Output only revised section content, not including titles
- Maintain original section objectives and positioning
- Use Markdown format for layout

Considerations:
- Don't deviate from section's basic objectives
- Don't introduce modifications unrelated to the issue
- Ensure completeness and readability of revised content"""
    
    def build_human_prompt(
        self,
        section: DocumentSection,
        issue_description: str,
        issue_suggestion: str,
        original_context: str,
        **kwargs: Any
    ) -> str:
        """Build human prompt for section revision."""
        if self.language == "zh":
            return f"""请修改以下章节内容以解决发现的问题：

=== 章节信息 ===
标题：{section.title}
目标：{section.goal}
层级：{section.level}

=== 当前内容 ===
{section.content}

=== 发现的问题 ===
问题描述：{issue_description}
修改建议：{issue_suggestion}

=== 原始背景信息 ===
{original_context}

修订指导：
- 重点解决上述问题
- 保持章节的基本目标和定位
- 确保修改后内容的质量和完整性
- 与原始背景信息保持一致

请输出修改后的章节内容（只输出正文，不包含标题）："""
        else:
            return f"""Please modify the following section content to resolve the identified issue:

=== Section Information ===
Title: {section.title}
Objective: {section.goal}
Level: {section.level}

=== Current Content ===
{section.content}

=== Identified Issue ===
Issue Description: {issue_description}
Modification Suggestion: {issue_suggestion}

=== Original Background Information ===
{original_context}

Revision Guidelines:
- Focus on resolving the above issue
- Maintain section's basic objectives and positioning
- Ensure quality and completeness of revised content
- Stay consistent with original background information

Please output the revised section content (body only, no title):"""
    
    def get_required_variables(self) -> List[str]:
        """Get required variables for section revision."""
        return ["section", "issue_description", "issue_suggestion", "original_context"]


class PromptManager:
    """Manages all prompts used in the system."""
    
    def __init__(self, language: str = "zh"):
        """Initialize prompt manager.
        
        Args:
            language: Default language for prompts
        """
        self.language = language
        self.builders = {
            PromptType.OUTLINE_GENERATION: OutlinePromptBuilder(language),
            PromptType.CONTENT_FILLING: ContentPromptBuilder(language),
            PromptType.DOCUMENT_REVIEW: ReviewPromptBuilder(language),
            PromptType.SECTION_REVISION: RevisionPromptBuilder(language)
        }
    
    def get_builder(self, prompt_type: PromptType) -> BasePromptBuilder:
        """Get prompt builder for the specified type."""
        return self.builders[prompt_type]
    
    def build_outline_prompts(self, context: str) -> tuple[str, str]:
        """Build system and human prompts for outline generation."""
        builder = self.get_builder(PromptType.OUTLINE_GENERATION)
        system_prompt = builder.build_system_prompt()
        human_prompt = builder.build_human_prompt(context=context)
        return system_prompt, human_prompt
    
    def build_content_prompts(
        self,
        section: DocumentSection,
        original_context: str,
        previous_content: str = "",
        document_title: str = ""
    ) -> tuple[str, str]:
        """Build system and human prompts for content filling."""
        builder = self.get_builder(PromptType.CONTENT_FILLING)
        system_prompt = builder.build_system_prompt()
        human_prompt = builder.build_human_prompt(
            section=section,
            original_context=original_context,
            previous_content=previous_content,
            document_title=document_title
        )
        return system_prompt, human_prompt
    
    def build_review_prompts(
        self,
        document_content: str,
        original_context: str
    ) -> tuple[str, str]:
        """Build system and human prompts for document review."""
        builder = self.get_builder(PromptType.DOCUMENT_REVIEW)
        system_prompt = builder.build_system_prompt()
        human_prompt = builder.build_human_prompt(
            document_content=document_content,
            original_context=original_context
        )
        return system_prompt, human_prompt
    
    def build_revision_prompts(
        self,
        section: DocumentSection,
        issue_description: str,
        issue_suggestion: str,
        original_context: str
    ) -> tuple[str, str]:
        """Build system and human prompts for section revision."""
        builder = self.get_builder(PromptType.SECTION_REVISION)
        system_prompt = builder.build_system_prompt()
        human_prompt = builder.build_human_prompt(
            section=section,
            issue_description=issue_description,
            issue_suggestion=issue_suggestion,
            original_context=original_context
        )
        return system_prompt, human_prompt