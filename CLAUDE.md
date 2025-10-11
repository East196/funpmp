# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**funpmp** 是一个基于AI智能体的项目管理模拟训练系统。它的核心是模拟真实项目环境下，人类项目经理与多个AI智能体协同工作的"训练场"。用户（项目经理）是大脑和指挥官，AI智能体是高效、专业但需要被管理的执行者。

## 核心架构

### 技术栈
- **Python 3.10+** 作为主要开发语言
- **LangGraph** 用于构建AI智能体工作流
- **LangChain** (OpenAI) 用于LLM集成
- **SQLModel** 用于数据建模
- **FastAPI** 用于API服务

### 项目结构
```
funpmp/
├── main.py              # 简单的入口文件
├── funpmp.py           # 核心AI智能体系统实现 (600+行)
├── ev.py               # 挣值管理计算函数
├── worktime.py         # 工期估算函数
├── test_ev.py          # 挣值管理测试用例
├── spec.md             # 详细的产品规格说明
├── requirements.txt    # 项目依赖
└── pyproject.toml     # 项目配置
```

### 核心组件
1. **ProjectManagementAgents**: AI智能体系统，包含：
   - 规划专家 (planner_agent)
   - 前端开发 (frontend_agent)
   - 后端开发 (backend_agent)
   - 项目经理 (manager_agent)

2. **ProjectState**: 项目状态管理，包含任务、决策请求、消息历史等

3. **ProjectManagerSimulator**: 项目管理模拟器，提供命令行交互界面

## 常用命令

### 环境管理
```bash
# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 运行应用
```bash
# 运行主程序（AI项目管理模拟器）
python funpmp.py

# 运行简单入口
python main.py

# 运行挣值管理测试
python test_ev.py
```

### 开发调试
```bash
# 运行挣值管理计算
python ev.py

# 运行工期估算计算
python worktime.py
```

## 核心业务逻辑

### 智能体工作流程
1. **规划阶段**: 规划专家生成WBS（工作分解结构）
2. **任务分配**: 项目经理分配任务给不同角色的AI智能体
3. **执行阶段**: AI智能体执行任务，会遇到各种"项目问题"
4. **决策阶段**: 当遇到问题时，AI智能体向人类项目经理请求决策
5. **协作阶段**: 不同AI智能体之间需要协作（如前后端接口约定）

### 关键数据结构
- **Task**: 任务对象，包含ID、标题、描述、状态、分配人等
- **TaskStatus**: 任务状态枚举（待办、进行中、已阻塞、已完成）
- **DecisionRequest**: 决策请求，包含问题描述和选项
- **AgentType**: AI智能体类型枚举

### 项目事件模拟
系统会模拟真实项目中的各种事件：
- **阻塞事件**: 技术不兼容、依赖未完成等
- **协作事件**: 前后端接口约定、方案评审等
- **进度事件**: 任务复杂度超出预期等

## 开发指南

### 代码规范
- 所有Python文件必须以 `# -*- coding: utf-8 -*-` 开头
- 数据结构尽可能使用强类型（dataclass、BaseModel等）
- 单个文件建议不超过300行（核心模块funpmp.py除外）
- 使用中文进行注释和文档

### 扩展智能体
要添加新的AI智能体角色：
1. 在AgentType枚举中添加新类型
2. 在ProjectManagementAgents中创建对应的agent方法
3. 在工作流中添加节点和路由
4. 为智能体定义专门的工具集

### 添加新工具
每个智能体都有专门的工具集，工具函数：
- 命名格式: `_action_tool`
- 接收字符串参数，返回字符串结果
- 在create_react_agent中注册

### 测试
- 挣值管理功能有独立测试用例test_ev.py
- 可以通过python test_ev.py运行测试
- 建议为核心功能添加更多单元测试

## 注意事项

- 需要设置OpenAI API密钥（在funpmp.py第18行）
- 项目目前处于MVP阶段，包含基础的AI智能体协作功能
- 命令行界面为临时实现，未来可能改为Web界面
- 所有交互和消息都是中文的