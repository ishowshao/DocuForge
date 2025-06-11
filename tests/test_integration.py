"""Integration tests for the complete DocuForge system."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from docuforge import (
    ClarificationItem,
    DocumentSection,
    DocumentStructure,
    RewriteRequest,
    create_rewrite_chain,
)


class MockLLM:
    """Mock LLM for integration testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    def invoke(self, messages, **kwargs):
        if self.call_count < len(self.responses):
            content = self.responses[self.call_count]
            self.call_count += 1
        else:
            content = "Default response"
        
        response = Mock()
        response.content = content
        return response


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""
    
    def test_complete_rewrite_workflow(self):
        """Test the complete rewrite workflow from request to result."""
        # Prepare LLM responses for the three-stage process
        outline_response = '''
        {
            "title": "企业级用户管理系统需求文档",
            "sections": [
                {
                    "title": "项目概述与目标",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "明确项目背景、核心目标和价值主张"
                },
                {
                    "title": "功能需求详述",
                    "content": "",
                    "level": 1,
                    "order": 1,
                    "goal": "详细描述各功能模块的具体需求和实现要点"
                },
                {
                    "title": "技术架构与实现",
                    "content": "",
                    "level": 1,
                    "order": 2,
                    "goal": "说明技术选型、架构设计和性能要求"
                },
                {
                    "title": "安全与合规要求",
                    "content": "",
                    "level": 1,
                    "order": 3,
                    "goal": "阐述数据安全、权限控制和合规标准"
                }
            ],
            "metadata": {"generated_by": "outline_generator"}
        }
        '''
        
        content_responses = [
            # 项目概述与目标
            """## 项目背景

企业级用户管理系统旨在为现代企业提供统一、安全、可扩展的用户身份管理解决方案。随着企业数字化转型的深入，传统的用户管理方式已无法满足复杂的业务需求。

## 核心目标

- **统一身份管理**：整合企业内部各系统的用户身份，实现单点登录
- **精细权限控制**：基于角色的权限管理，支持多层级权限分配
- **高安全标准**：符合SOC2等安全合规要求，保障数据安全
- **高可用架构**：支持1000-5000并发用户，峰值10000用户的高性能需求

## 价值主张

通过构建企业级用户管理系统，企业可以显著提升用户体验、降低管理成本、增强数据安全，为数字化转型奠定坚实基础。""",
            
            # 功能需求详述
            """## 用户注册与认证

### 用户注册
- **必填字段**：用户名、邮箱、密码、手机号码、公司名称
- **验证机制**：邮箱验证、手机短信验证
- **密码策略**：强密码要求，包含大小写字母、数字和特殊字符

### 用户登录
- **多种登录方式**：用户名/邮箱登录、手机号登录
- **安全验证**：支持双因子认证（2FA）
- **会话管理**：智能会话超时和刷新机制

## 权限管理系统

### 角色层级
1. **超级管理员**：系统全权管理，用户管理，权限分配
2. **部门管理员**：部门内用户管理，部门权限分配
3. **普通用户**：基础功能使用，个人信息管理

### 权限控制
- **基于角色的访问控制（RBAC）**
- **细粒度权限分配**：功能级、数据级权限控制
- **权限继承与覆盖**：灵活的权限传递机制

## 第三方集成

### LDAP集成
- 支持企业现有LDAP系统集成
- 用户信息自动同步
- 统一认证流程

### SSO单点登录
- SAML 2.0协议支持
- OAuth 2.0授权流程
- 跨系统无缝切换

### ERP系统集成
- RESTful API接口
- 实时数据同步
- 业务流程整合""",
            
            # 技术架构与实现
            """## 技术选型

### 后端技术栈
- **框架**：Spring Boot 2.7+ / Django 4.0+
- **数据库**：PostgreSQL（主库）+ Redis（缓存）
- **消息队列**：RabbitMQ / Apache Kafka
- **搜索引擎**：Elasticsearch

### 前端技术栈
- **框架**：React 18+ / Vue 3+
- **UI组件库**：Ant Design / Element Plus
- **状态管理**：Redux Toolkit / Pinia

## 架构设计

### 微服务架构
- **用户服务**：用户信息管理、认证授权
- **权限服务**：角色权限管理、访问控制
- **集成服务**：第三方系统集成、数据同步
- **通知服务**：邮件、短信、系统通知

### 性能设计
- **并发处理**：支持1000-5000并发用户
- **峰值处理**：弹性扩缩容，峰值10000用户
- **响应时间**：API响应时间 < 200ms
- **可用性**：99.9%系统可用性保障

## 部署架构

### 容器化部署
- **Docker容器化**：统一运行环境
- **Kubernetes编排**：自动化部署和扩缩容
- **CI/CD流水线**：自动化测试和部署

### 监控运维
- **应用监控**：APM性能监控
- **日志管理**：集中化日志收集和分析
- **告警机制**：多维度监控告警""",
            
            # 安全与合规要求
            """## 数据安全

### 数据加密
- **传输加密**：HTTPS/TLS 1.3全链路加密
- **存储加密**：敏感数据AES-256加密存储
- **密钥管理**：专业密钥管理系统（KMS）

### 访问控制
- **最小权限原则**：用户仅获得必要的最小权限
- **访问审计**：详细的用户行为日志记录
- **异常检测**：自动检测异常登录和操作行为

## 操作审计

### 审计日志
- **操作记录**：用户所有操作的详细日志
- **数据变更**：数据修改前后的完整记录
- **登录轨迹**：用户登录时间、地点、设备信息

### 审计报告
- **定期审计**：月度、季度安全审计报告
- **合规检查**：SOC2标准合规性检查
- **风险评估**：定期安全风险评估和改进建议

## 合规标准

### SOC2合规
- **安全性**：数据保护和访问控制
- **可用性**：系统稳定性和业务连续性
- **处理完整性**：数据处理的准确性和完整性
- **保密性**：敏感信息的保护措施
- **隐私性**：个人信息的合规处理

### 其他合规要求
- **GDPR合规**：欧盟数据保护法规
- **ISO 27001**：信息安全管理体系
- **等保三级**：国家信息安全等级保护"""
        ]
        
        # No issues in review
        review_response = '''
        {
            "issues": [],
            "overall_quality": "文档结构清晰，内容完整，涵盖了企业级用户管理系统的各个重要方面。技术方案可行，安全要求符合标准。"
        }
        '''
        
        # Setup mock LLM with all responses
        responses = [outline_response] + content_responses + [review_response]
        mock_llm = MockLLM(responses)
        
        # Create comprehensive request
        request = RewriteRequest(
            original_content="""
# 产品需求文档

## 项目概述

我们需要开发一个新的用户管理系统，用于处理企业级用户的注册、认证和权限管理。

## 功能需求

系统应该包含以下功能：
- 用户注册和登录
- 权限管理
- 数据安全

## 技术要求

系统需要具备高可用性和可扩展性。
            """.strip(),
            clarifications=[
                ClarificationItem(
                    question="用户注册需要哪些必填字段？",
                    answer="用户名、邮箱、密码、手机号码、公司名称是必填字段"
                ),
                ClarificationItem(
                    question="权限管理的具体层级是什么？",
                    answer="分为超级管理员、部门管理员、普通用户三个层级，支持基于角色的权限控制"
                ),
                ClarificationItem(
                    question="对数据安全有什么特殊要求？",
                    answer="需要支持数据加密存储、操作日志记录、定期安全审计，符合SOC2标准"
                ),
                ClarificationItem(
                    question="系统的并发用户数预期是多少？",
                    answer="预期支持1000-5000并发用户，峰值可能达到10000用户"
                ),
                ClarificationItem(
                    question="是否需要支持第三方集成？",
                    answer="需要支持LDAP、SSO单点登录，以及与现有ERP系统的API集成"
                )
            ]
        )
        
        # Create and execute rewrite chain
        chain = create_rewrite_chain(llm=mock_llm, max_revision_rounds=2)
        result = chain.invoke(request)
        
        # Verify result structure
        assert result.rewritten_content is not None
        assert isinstance(result.structured_document, DocumentStructure)
        
        # Verify document structure
        doc = result.structured_document
        assert doc.title == "企业级用户管理系统需求文档"
        assert len(doc.sections) == 4
        
        # Verify sections have content
        section_titles = [section.title for section in doc.sections]
        expected_titles = [
            "项目概述与目标",
            "功能需求详述", 
            "技术架构与实现",
            "安全与合规要求"
        ]
        assert section_titles == expected_titles
        
        # Verify all sections have content filled
        for section in doc.sections:
            assert section.content.strip() != ""
            assert len(section.content) > 100  # Substantial content
        
        # Verify markdown output contains key information
        markdown = result.rewritten_content
        assert "企业级用户管理系统需求文档" in markdown
        assert "项目概述与目标" in markdown
        assert "功能需求详述" in markdown
        assert "SOC2" in markdown  # From clarifications
        assert "1000-5000并发用户" in markdown  # From clarifications
        
        # Verify all clarification information was incorporated
        for clarification in request.clarifications:
            # Key terms from clarifications should appear in the final document
            if "必填字段" in clarification.answer:
                assert "用户名、邮箱、密码" in markdown
            elif "超级管理员" in clarification.answer:
                assert "超级管理员" in markdown
            elif "SOC2" in clarification.answer:
                assert "SOC2" in markdown
    
    def test_workflow_with_revision_cycle(self):
        """Test workflow that includes revision cycle."""
        outline_response = '''
        {
            "title": "测试文档",
            "sections": [
                {
                    "title": "概述",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "提供项目概述"
                }
            ],
            "metadata": {}
        }
        '''
        
        initial_content = "这是初始内容，存在一些问题。"
        
        # Review with issues
        review_with_issues = '''
        {
            "issues": [
                {
                    "section_order": 0,
                    "issue_type": "quality_issue",
                    "description": "内容太简单，需要更详细的描述",
                    "suggestion": "增加更多技术细节和实施方案"
                }
            ],
            "overall_quality": "内容需要进一步完善"
        }
        '''
        
        revised_content = "这是修订后的详细内容，包含了技术细节和实施方案，更加完整和专业。"
        
        # Final review (no issues)
        final_review = '''
        {
            "issues": [],
            "overall_quality": "内容完整，质量良好"
        }
        '''
        
        responses = [
            outline_response,      # Outline generation
            initial_content,       # Initial content filling
            review_with_issues,    # First review (has issues)
            revised_content,       # Revision
            final_review          # Final review (no issues)
        ]
        
        mock_llm = MockLLM(responses)
        
        request = RewriteRequest(
            original_content="测试原始内容",
            clarifications=[
                ClarificationItem(question="需要什么？", answer="需要详细的技术方案")
            ]
        )
        
        chain = create_rewrite_chain(llm=mock_llm, max_revision_rounds=2)
        result = chain.invoke(request)
        
        # Should have gone through revision cycle
        assert result.structured_document.sections[0].content == revised_content
        assert "技术细节和实施方案" in result.rewritten_content
        
        # Should have made 5 LLM calls (outline + content + review + revision + final review)
        assert mock_llm.call_count == 5
    
    def test_error_handling_in_workflow(self):
        """Test error handling in the complete workflow."""
        # Invalid outline JSON
        invalid_outline = "这不是有效的JSON格式"
        
        mock_llm = MockLLM([invalid_outline])
        
        request = RewriteRequest(
            original_content="测试内容",
            clarifications=[]
        )
        
        chain = create_rewrite_chain(llm=mock_llm)
        
        # Should raise ValueError due to invalid JSON
        with pytest.raises(ValueError, match="Workflow failed"):
            chain.invoke(request)


class TestCLIIntegration:
    """Test CLI integration with the complete system."""
    
    @patch('docuforge.cli.setup_llm')
    def test_cli_integration_flow(self, mock_setup_llm):
        """Test CLI integration with file I/O."""
        from docuforge.cli import (
            load_clarifications_from_file,
            load_document_from_file,
            save_output_file
        )
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create document file
            doc_file = os.path.join(temp_dir, "document.md")
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write("""# 系统需求文档

## 概述
需要开发一个新系统。

## 需求
- 功能A
- 功能B
""")
            
            # Create clarifications file
            clarify_file = os.path.join(temp_dir, "clarifications.json")
            clarifications_data = [
                {
                    "question": "系统的主要用途是什么？",
                    "answer": "用于企业内部管理"
                },
                {
                    "question": "需要支持多少用户？",
                    "answer": "预计100-500个用户"
                }
            ]
            with open(clarify_file, 'w', encoding='utf-8') as f:
                json.dump(clarifications_data, f, ensure_ascii=False, indent=2)
            
            # Test file loading
            document_content = load_document_from_file(doc_file)
            assert "系统需求文档" in document_content
            assert "需要开发一个新系统" in document_content
            
            clarifications = load_clarifications_from_file(clarify_file)
            assert len(clarifications) == 2
            assert clarifications[0].question == "系统的主要用途是什么？"
            assert clarifications[0].answer == "用于企业内部管理"
            
            # Test output saving
            output_file = os.path.join(temp_dir, "output.md")
            test_content = "# 输出测试\n\n这是测试输出内容。"
            save_output_file(test_content, output_file)
            
            # Verify output file
            assert os.path.exists(output_file)
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            assert saved_content == test_content


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_complex_document_structure(self):
        """Test handling complex document with multiple levels."""
        outline_response = '''
        {
            "title": "复杂系统架构文档",
            "sections": [
                {
                    "title": "系统概述",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "介绍系统整体架构"
                },
                {
                    "title": "前端架构",
                    "content": "",
                    "level": 2,
                    "order": 1,
                    "goal": "详述前端技术方案"
                },
                {
                    "title": "用户界面设计",
                    "content": "",
                    "level": 3,
                    "order": 2,
                    "goal": "说明UI/UX设计规范"
                },
                {
                    "title": "后端架构",
                    "content": "",
                    "level": 2,
                    "order": 3,
                    "goal": "详述后端技术方案"
                },
                {
                    "title": "数据库设计",
                    "content": "",
                    "level": 3,
                    "order": 4,
                    "goal": "说明数据模型和存储方案"
                }
            ],
            "metadata": {}
        }
        '''
        
        content_responses = [
            "系统采用微服务架构，前后端分离设计...",
            "前端使用React框架，配合TypeScript开发...",
            "遵循Material Design设计语言，注重用户体验...", 
            "后端基于Spring Boot微服务架构...",
            "采用PostgreSQL主数据库，Redis作为缓存层..."
        ]
        
        review_response = '''
        {
            "issues": [],
            "overall_quality": "文档结构层次清晰，技术方案合理"
        }
        '''
        
        responses = [outline_response] + content_responses + [review_response]
        mock_llm = MockLLM(responses)
        
        request = RewriteRequest(
            original_content="需要设计一个复杂的系统架构",
            clarifications=[
                ClarificationItem(
                    question="前端使用什么技术？",
                    answer="React + TypeScript"
                ),
                ClarificationItem(
                    question="数据库选型？",
                    answer="PostgreSQL + Redis"
                )
            ]
        )
        
        chain = create_rewrite_chain(llm=mock_llm)
        result = chain.invoke(request)
        
        # Verify complex structure
        doc = result.structured_document
        assert len(doc.sections) == 5
        
        # Verify hierarchical levels
        levels = [section.level for section in doc.sections]
        assert levels == [1, 2, 3, 2, 3]
        
        # Verify content hierarchy in markdown
        markdown = result.rewritten_content
        assert "# 复杂系统架构文档" in markdown
        assert "## 系统概述" in markdown
        assert "### 前端架构" in markdown
        assert "#### 用户界面设计" in markdown
        assert "### 后端架构" in markdown
        assert "#### 数据库设计" in markdown
    
    def test_multilingual_content_handling(self):
        """Test handling of multilingual content."""
        # Test with Chinese content and English technical terms
        outline_response = '''
        {
            "title": "Multi-language System Documentation",
            "sections": [
                {
                    "title": "系统介绍 (System Introduction)",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "Introduce the multilingual system"
                }
            ],
            "metadata": {}
        }
        '''
        
        content_response = """## 系统特性 (System Features)

本系统支持多语言环境，包含以下特性：

### 技术特性 (Technical Features)
- **RESTful API**: 符合REST架构风格的API设计
- **i18n Support**: 国际化支持，支持中英文切换
- **Unicode Compatibility**: 完整的Unicode字符支持

### 业务特性 (Business Features)
- **Multi-tenant**: 多租户架构支持
- **Real-time Processing**: 实时数据处理能力
- **Cloud Native**: 云原生架构设计"""
        
        review_response = '''
        {
            "issues": [],
            "overall_quality": "Multilingual content handled properly with good structure"
        }
        '''
        
        responses = [outline_response, content_response, review_response]
        mock_llm = MockLLM(responses)
        
        request = RewriteRequest(
            original_content="需要一个支持多语言的系统 (Need a multilingual system)",
            clarifications=[
                ClarificationItem(
                    question="支持哪些语言？(Which languages to support?)",
                    answer="中文和English"
                )
            ]
        )
        
        chain = create_rewrite_chain(llm=mock_llm)
        result = chain.invoke(request)
        
        # Verify multilingual content is preserved
        markdown = result.rewritten_content
        assert "系统介绍 (System Introduction)" in markdown
        assert "RESTful API" in markdown
        assert "中英文切换" in markdown
        assert "Multi-tenant" in markdown