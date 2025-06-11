"""Unit tests for prompt system."""

import pytest

from docuforge.prompts import (
    BasePromptBuilder,
    OutlinePromptBuilder,
    ContentPromptBuilder,
    ReviewPromptBuilder,
    RevisionPromptBuilder,
    PromptManager,
    PromptType,
    PromptConfig,
)
from docuforge.models import DocumentSection
from docuforge.components import RevisionIssue


class TestPromptConfig:
    """Test PromptConfig model."""
    
    def test_valid_creation(self):
        """Test creating a valid PromptConfig."""
        config = PromptConfig(
            name="test_prompt",
            version="1.0",
            prompt_type=PromptType.OUTLINE_GENERATION,
            template="Test template with {variable}",
            variables=["variable"],
            description="Test prompt for testing"
        )
        
        assert config.name == "test_prompt"
        assert config.version == "1.0"
        assert config.prompt_type == PromptType.OUTLINE_GENERATION
        assert "variable" in config.variables


class TestOutlinePromptBuilder:
    """Test OutlinePromptBuilder."""
    
    def test_chinese_system_prompt(self):
        """Test Chinese system prompt generation."""
        builder = OutlinePromptBuilder(language="zh")
        system_prompt = builder.build_system_prompt()
        
        assert "资深的产品经理" in system_prompt
        assert "JSON格式" in system_prompt
        assert "content字段必须为空字符串" in system_prompt
        assert "goal字段要具体描述" in system_prompt
    
    def test_english_system_prompt(self):
        """Test English system prompt generation."""
        builder = OutlinePromptBuilder(language="en")
        system_prompt = builder.build_system_prompt()
        
        assert "senior product manager" in system_prompt
        assert "JSON format" in system_prompt
        assert "content field must be empty string" in system_prompt
        assert "goal field should specifically describe" in system_prompt
    
    def test_chinese_human_prompt(self):
        """Test Chinese human prompt generation."""
        builder = OutlinePromptBuilder(language="zh")
        context = "这是测试上下文内容"
        
        human_prompt = builder.build_human_prompt(context=context)
        
        assert "这是测试上下文内容" in human_prompt
        assert "请基于以下背景信息" in human_prompt
        assert "设计一个清晰的文档结构大纲" in human_prompt
    
    def test_english_human_prompt(self):
        """Test English human prompt generation."""
        builder = OutlinePromptBuilder(language="en")
        context = "This is test context content"
        
        human_prompt = builder.build_human_prompt(context=context)
        
        assert "This is test context content" in human_prompt
        assert "design a clear document structure outline" in human_prompt
    
    def test_required_variables(self):
        """Test required variables for outline generation."""
        builder = OutlinePromptBuilder()
        variables = builder.get_required_variables()
        
        assert variables == ["context"]


class TestContentPromptBuilder:
    """Test ContentPromptBuilder."""
    
    def test_chinese_system_prompt(self):
        """Test Chinese system prompt for content filling."""
        builder = ContentPromptBuilder(language="zh")
        system_prompt = builder.build_system_prompt()
        
        assert "专业的技术文档撰写专家" in system_prompt
        assert "目标导向" in system_prompt
        assert "上下文连贯" in system_prompt
        assert "只输出章节正文内容" in system_prompt
    
    def test_english_system_prompt(self):
        """Test English system prompt for content filling."""
        builder = ContentPromptBuilder(language="en")
        system_prompt = builder.build_system_prompt()
        
        assert "professional technical documentation expert" in system_prompt
        assert "Goal-Oriented" in system_prompt
        assert "Contextual Coherence" in system_prompt
        assert "Output only section body content" in system_prompt
    
    def test_chinese_human_prompt_with_previous_content(self):
        """Test Chinese human prompt with previous content."""
        builder = ContentPromptBuilder(language="zh")
        
        section = DocumentSection(
            title="测试章节",
            content="",
            level=2,
            order=1,
            goal="测试内容生成"
        )
        
        human_prompt = builder.build_human_prompt(
            section=section,
            original_context="原始背景信息",
            previous_content="前面章节的内容",
            document_title="测试文档"
        )
        
        assert "文档标题：测试文档" in human_prompt
        assert "当前章节：测试章节" in human_prompt
        assert "章节目标：测试内容生成" in human_prompt
        assert "章节层级：2" in human_prompt
        assert "章节顺序：1" in human_prompt
        assert "原始背景信息" in human_prompt
        assert "前面章节的内容" in human_prompt
    
    def test_chinese_human_prompt_without_previous_content(self):
        """Test Chinese human prompt without previous content."""
        builder = ContentPromptBuilder(language="zh")
        
        section = DocumentSection(
            title="首个章节",
            content="",
            level=1,
            order=0,
            goal="开始文档"
        )
        
        human_prompt = builder.build_human_prompt(
            section=section,
            original_context="背景信息",
            document_title="文档标题"
        )
        
        assert "当前章节：首个章节" in human_prompt
        assert "前面章节的内容" not in human_prompt  # Should not appear for first section
    
    def test_required_variables(self):
        """Test required variables for content filling."""
        builder = ContentPromptBuilder()
        variables = builder.get_required_variables()
        
        expected = ["section", "original_context", "document_title"]
        assert variables == expected


class TestReviewPromptBuilder:
    """Test ReviewPromptBuilder."""
    
    def test_chinese_system_prompt(self):
        """Test Chinese system prompt for document review."""
        builder = ReviewPromptBuilder(language="zh")
        system_prompt = builder.build_system_prompt()
        
        assert "专业的文档评审专家" in system_prompt
        assert "逻辑一致性" in system_prompt
        assert "信息完整性" in system_prompt
        assert "结构合理性" in system_prompt
        assert "内容准确性" in system_prompt
        assert "RevisionReport对象" in system_prompt
        assert "issues数组应为空列表" in system_prompt
    
    def test_english_system_prompt(self):
        """Test English system prompt for document review."""
        builder = ReviewPromptBuilder(language="en")
        system_prompt = builder.build_system_prompt()
        
        assert "professional document review expert" in system_prompt
        assert "Logical Consistency" in system_prompt
        assert "Information Completeness" in system_prompt
        assert "Structural Reasonableness" in system_prompt
        assert "Content Accuracy" in system_prompt
        assert "RevisionReport object" in system_prompt
        assert "issues array should be an empty list" in system_prompt
    
    def test_chinese_human_prompt(self):
        """Test Chinese human prompt for document review."""
        builder = ReviewPromptBuilder(language="zh")
        
        human_prompt = builder.build_human_prompt(
            document_content="生成的文档内容",
            original_context="原始需求背景"
        )
        
        assert "原始需求和背景" in human_prompt
        assert "原始需求背景" in human_prompt
        assert "生成的文档内容" in human_prompt
        assert "识别可能存在的问题" in human_prompt
    
    def test_required_variables(self):
        """Test required variables for document review."""
        builder = ReviewPromptBuilder()
        variables = builder.get_required_variables()
        
        expected = ["document_content", "original_context"]
        assert variables == expected


class TestRevisionPromptBuilder:
    """Test RevisionPromptBuilder."""
    
    def test_chinese_system_prompt(self):
        """Test Chinese system prompt for section revision."""
        builder = RevisionPromptBuilder(language="zh")
        system_prompt = builder.build_system_prompt()
        
        assert "专业的文档修订专家" in system_prompt
        assert "精准修改" in system_prompt
        assert "问题导向" in system_prompt
        assert "保持一致性" in system_prompt
        assert "只输出修改后的章节内容" in system_prompt
    
    def test_english_system_prompt(self):
        """Test English system prompt for section revision."""
        builder = RevisionPromptBuilder(language="en")
        system_prompt = builder.build_system_prompt()
        
        assert "professional document revision expert" in system_prompt
        assert "Precise Modification" in system_prompt
        assert "Problem-Oriented" in system_prompt
        assert "Maintain Consistency" in system_prompt
        assert "Output only revised section content" in system_prompt
    
    def test_chinese_human_prompt(self):
        """Test Chinese human prompt for section revision."""
        builder = RevisionPromptBuilder(language="zh")
        
        section = DocumentSection(
            title="需要修改的章节",
            content="当前有问题的内容",
            level=1,
            order=0,
            goal="章节目标"
        )
        
        human_prompt = builder.build_human_prompt(
            section=section,
            issue_description="发现的问题描述",
            issue_suggestion="修改建议",
            original_context="原始背景"
        )
        
        assert "标题：需要修改的章节" in human_prompt
        assert "目标：章节目标" in human_prompt
        assert "当前有问题的内容" in human_prompt
        assert "问题描述：发现的问题描述" in human_prompt
        assert "修改建议：修改建议" in human_prompt
        assert "原始背景" in human_prompt
    
    def test_required_variables(self):
        """Test required variables for section revision."""
        builder = RevisionPromptBuilder()
        variables = builder.get_required_variables()
        
        expected = ["section", "issue_description", "issue_suggestion", "original_context"]
        assert variables == expected


class TestPromptManager:
    """Test PromptManager."""
    
    def test_initialization_default_language(self):
        """Test PromptManager initialization with default language."""
        manager = PromptManager()
        
        assert manager.language == "zh"
        assert PromptType.OUTLINE_GENERATION in manager.builders
        assert PromptType.CONTENT_FILLING in manager.builders
        assert PromptType.DOCUMENT_REVIEW in manager.builders
        assert PromptType.SECTION_REVISION in manager.builders
    
    def test_initialization_custom_language(self):
        """Test PromptManager initialization with custom language."""
        manager = PromptManager(language="en")
        
        assert manager.language == "en"
    
    def test_get_builder(self):
        """Test getting specific prompt builder."""
        manager = PromptManager()
        
        outline_builder = manager.get_builder(PromptType.OUTLINE_GENERATION)
        assert isinstance(outline_builder, OutlinePromptBuilder)
        
        content_builder = manager.get_builder(PromptType.CONTENT_FILLING)
        assert isinstance(content_builder, ContentPromptBuilder)
        
        review_builder = manager.get_builder(PromptType.DOCUMENT_REVIEW)
        assert isinstance(review_builder, ReviewPromptBuilder)
        
        revision_builder = manager.get_builder(PromptType.SECTION_REVISION)
        assert isinstance(revision_builder, RevisionPromptBuilder)
    
    def test_build_outline_prompts(self):
        """Test building outline prompts."""
        manager = PromptManager(language="zh")
        context = "测试上下文"
        
        system_prompt, human_prompt = manager.build_outline_prompts(context)
        
        assert "资深的产品经理" in system_prompt
        assert "测试上下文" in human_prompt
        assert isinstance(system_prompt, str)
        assert isinstance(human_prompt, str)
    
    def test_build_content_prompts(self):
        """Test building content prompts."""
        manager = PromptManager(language="zh")
        
        section = DocumentSection(
            title="测试章节",
            content="",
            level=1,
            order=0,
            goal="测试目标"
        )
        
        system_prompt, human_prompt = manager.build_content_prompts(
            section=section,
            original_context="原始上下文",
            previous_content="前面内容",
            document_title="文档标题"
        )
        
        assert "技术文档撰写专家" in system_prompt
        assert "测试章节" in human_prompt
        assert "原始上下文" in human_prompt
        assert "前面内容" in human_prompt
    
    def test_build_review_prompts(self):
        """Test building review prompts."""
        manager = PromptManager(language="zh")
        
        system_prompt, human_prompt = manager.build_review_prompts(
            document_content="文档内容",
            original_context="原始上下文"
        )
        
        assert "文档评审专家" in system_prompt
        assert "文档内容" in human_prompt
        assert "原始上下文" in human_prompt
    
    def test_build_revision_prompts(self):
        """Test building revision prompts."""
        manager = PromptManager(language="zh")
        
        section = DocumentSection(
            title="测试章节",
            content="需要修改的内容",
            level=1,
            order=0,
            goal="目标"
        )
        
        system_prompt, human_prompt = manager.build_revision_prompts(
            section=section,
            issue_description="问题描述",
            issue_suggestion="修改建议",
            original_context="原始上下文"
        )
        
        assert "文档修订专家" in system_prompt
        assert "测试章节" in human_prompt
        assert "问题描述" in human_prompt
        assert "修改建议" in human_prompt


class TestPromptType:
    """Test PromptType enum."""
    
    def test_prompt_type_values(self):
        """Test PromptType enum values."""
        assert PromptType.OUTLINE_GENERATION == "outline_generation"
        assert PromptType.CONTENT_FILLING == "content_filling"
        assert PromptType.DOCUMENT_REVIEW == "document_review"
        assert PromptType.SECTION_REVISION == "section_revision"
    
    def test_prompt_type_enum_membership(self):
        """Test PromptType enum membership."""
        all_types = list(PromptType)
        assert len(all_types) == 4
        assert PromptType.OUTLINE_GENERATION in all_types
        assert PromptType.CONTENT_FILLING in all_types
        assert PromptType.DOCUMENT_REVIEW in all_types
        assert PromptType.SECTION_REVISION in all_types


class TestPromptIntegration:
    """Integration tests for prompt system."""
    
    def test_multilingual_consistency(self):
        """Test that Chinese and English prompts cover the same functionality."""
        # Test outline generation
        zh_builder = OutlinePromptBuilder(language="zh")
        en_builder = OutlinePromptBuilder(language="en")
        
        zh_system = zh_builder.build_system_prompt()
        en_system = en_builder.build_system_prompt()
        
        # Both should mention JSON format
        assert "JSON" in zh_system
        assert "JSON" in en_system
        
        # Both should have similar structure requirements
        assert "content" in zh_system and "content" in en_system
        assert "goal" in zh_system and "goal" in en_system
    
    def test_prompt_builder_inheritance(self):
        """Test that all prompt builders inherit from BasePromptBuilder."""
        builders = [
            OutlinePromptBuilder(),
            ContentPromptBuilder(),
            ReviewPromptBuilder(),
            RevisionPromptBuilder()
        ]
        
        for builder in builders:
            assert isinstance(builder, BasePromptBuilder)
            # Each builder should implement required methods
            assert hasattr(builder, 'build_system_prompt')
            assert hasattr(builder, 'build_human_prompt')
            assert hasattr(builder, 'get_required_variables')
    
    def test_context_preservation(self):
        """Test that context is properly preserved through prompt building."""
        manager = PromptManager()
        test_context = "重要的测试上下文信息，包含特殊字符：@#$%^&*()"
        
        # Test outline prompts
        _, human_prompt = manager.build_outline_prompts(test_context)
        assert test_context in human_prompt
        
        # Test review prompts
        _, human_prompt = manager.build_review_prompts(
            document_content="文档内容",
            original_context=test_context
        )
        assert test_context in human_prompt
    
    def test_section_data_preservation(self):
        """Test that section data is properly preserved in prompts."""
        manager = PromptManager()
        
        section = DocumentSection(
            title="特殊章节标题 @#$",
            content="特殊内容 %^&*()",
            level=3,
            order=5,
            goal="特殊目标描述"
        )
        
        _, human_prompt = manager.build_content_prompts(
            section=section,
            original_context="上下文",
            document_title="文档标题"
        )
        
        assert "特殊章节标题 @#$" in human_prompt
        assert "特殊目标描述" in human_prompt
        assert "层级：3" in human_prompt
        assert "顺序：5" in human_prompt