#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
from typing import Dict, List, Any, Optional, Annotated
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from pydantic import BaseModel, Field

# 自动加载环境变量
load_dotenv()


class TaskStatus(Enum):
    TODO = "待办"
    IN_PROGRESS = "进行中"
    BLOCKED = "已阻塞"
    COMPLETED = "已完成"

class AgentType(Enum):
    PLANNER = "规划专家"
    FRONTEND_DEV = "前端开发"
    BACKEND_DEV = "后端开发"
    PROJECT_MANAGER = "项目经理"

@dataclass
class Task:
    id: int
    title: str
    description: str
    status: TaskStatus
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class DecisionRequest:
    task_id: int
    from_agent: str
    problem: str
    options: List[str]
    selected_option: Optional[int] = None

class ProjectState(BaseModel):
    """项目状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    project_goal: str = ""
    tasks: List[Task] = Field(default_factory=list)
    decision_requests: List[DecisionRequest] = Field(default_factory=list)
    current_agent: AgentType = AgentType.PROJECT_MANAGER
    api_draft: str = ""
    completed: bool = False
    current_task_id: Optional[int] = None

class ProjectManagementAgents:
    """项目管理智能体系统"""

    def __init__(self):
        # 从环境变量读取配置
        self._load_config()

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base
        )

        # 创建各角色智能体
        self.planner_agent = self._create_planner_agent()
        self.frontend_agent = self._create_frontend_agent()
        self.backend_agent = self._create_backend_agent()
        self.manager_agent = self._create_manager_agent()

        # 构建智能体工作流
        self.workflow = self._build_workflow()

    def _load_config(self):
        """从环境变量加载配置"""
        # API密钥配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key == "your-openai-api-key-here":
            print("警告：未设置有效的OpenAI API密钥")
            print("请在.env文件中设置 OPENAI_API_KEY 或设置环境变量")
            # 为了演示目的使用一个测试密钥
            self.api_key = "sk-test-key-for-demo"

        # API基础URL（可选）
        self.api_base = os.getenv("OPENAI_API_BASE")

        # 模型配置（可选）
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # 温度参数（可选）
        temp_str = os.getenv("LLM_TEMPERATURE", "0.7")
        try:
            self.temperature = float(temp_str)
        except ValueError:
            self.temperature = 0.7
            print(f"警告：无效的温度参数 {temp_str}，使用默认值 0.7")
    
    def _create_planner_agent(self):
        """创建规划专家智能体"""
        tools = [
            Tool(
                name="generate_wbs",
                func=self._generate_wbs_tool,
                description="根据项目目标生成工作分解结构(WBS)"
            ),
            Tool(
                name="update_task_status",
                func=self._update_task_status_tool,
                description="更新任务状态"
            )
        ]
        return create_react_agent(self.llm, tools)
    
    def _create_frontend_agent(self):
        """创建前端开发智能体"""
        tools = [
            Tool(
                name="work_on_task",
                func=self._work_on_task_tool,
                description="执行前端开发任务"
            ),
            Tool(
                name="review_api_draft",
                func=self._review_api_draft_tool,
                description="评审API草案并提出问题"
            ),
            Tool(
                name="report_problem",
                func=self._report_problem_tool,
                description="报告遇到的问题"
            )
        ]
        return create_react_agent(self.llm, tools)
    
    def _create_backend_agent(self):
        """创建后端开发智能体"""
        tools = [
            Tool(
                name="work_on_task",
                func=self._work_on_task_tool,
                description="执行后端开发任务"
            ),
            Tool(
                name="create_api_draft",
                func=self._create_api_draft_tool,
                description="创建API接口草案"
            ),
            Tool(
                name="report_problem",
                func=self._report_problem_tool,
                description="报告遇到的问题"
            )
        ]
        return create_react_agent(self.llm, tools)
    
    def _create_manager_agent(self):
        """创建项目经理智能体"""
        tools = [
            Tool(
                name="assign_tasks",
                func=self._assign_tasks_tool,
                description="分配任务给团队成员"
            ),
            Tool(
                name="make_decision",
                func=self._make_decision_tool,
                description="对问题做出决策"
            ),
            Tool(
                name="check_progress",
                func=self._check_progress_tool,
                description="检查项目进度"
            )
        ]
        return create_react_agent(self.llm, tools)
    
    def _build_workflow(self):
        """构建智能体工作流"""
        workflow = StateGraph(ProjectState)
        
        # 添加节点
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("frontend", self._frontend_node)
        workflow.add_node("backend", self._backend_node)
        workflow.add_node("manager", self._manager_node)
        workflow.add_node("human_input", self._human_input_node)
        
        # 设置入口点
        workflow.set_entry_point("manager")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "manager",
            self._route_after_manager,
            {
                "planner": "planner",
                "frontend": "frontend", 
                "backend": "backend",
                "human_input": "human_input",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "manager": "manager",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "frontend",
            self._route_after_developer,
            {
                "manager": "manager",
                "backend": "backend",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "backend", 
            self._route_after_developer,
            {
                "manager": "manager",
                "frontend": "frontend",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "human_input",
            self._route_after_human,
            {
                "manager": "manager",
                "end": END
            }
        )
        
        return workflow.compile()

    def _reconstruct_state(self, result_dict):
        """重建状态对象的辅助方法"""
        from copy import deepcopy

        # 重建任务对象
        tasks = []
        for task_data in result_dict.get('tasks', []):
            if isinstance(task_data, dict):
                tasks.append(Task(
                    id=task_data['id'],
                    title=task_data['title'],
                    description=task_data['description'],
                    status=TaskStatus(task_data['status']),
                    assigned_to=task_data['assigned_to'],
                    created_at=task_data['created_at'],
                    updated_at=task_data['updated_at']
                ))
            else:
                tasks.append(task_data)

        # 重建决策请求对象
        decision_requests = []
        for dr_data in result_dict.get('decision_requests', []):
            if isinstance(dr_data, dict):
                decision_requests.append(DecisionRequest(
                    task_id=dr_data['task_id'],
                    from_agent=dr_data['from_agent'],
                    problem=dr_data['problem'],
                    options=dr_data['options'],
                    selected_option=dr_data['selected_option']
                ))
            else:
                decision_requests.append(dr_data)

        return ProjectState(
            messages=result_dict.get('messages', []),
            project_goal=result_dict.get('project_goal', ''),
            tasks=tasks,
            decision_requests=decision_requests,
            current_agent=AgentType(result_dict.get('current_agent', AgentType.PROJECT_MANAGER)),
            api_draft=result_dict.get('api_draft', ''),
            completed=result_dict.get('completed', False),
            current_task_id=result_dict.get('current_task_id')
        )
    
    # 工具函数
    def _generate_wbs_tool(self, project_goal: str) -> str:
        """生成工作分解结构"""
        # 在实际应用中，这里会调用LLM生成WBS
        # 这里简化为固定模板
        wbs = [
            {"id": 1, "title": "数据库设计", "description": "创建用户表，设计字段"},
            {"id": 2, "title": "后端API开发", "description": "开发登录接口和登出接口"},
            {"id": 3, "title": "前端页面开发", "description": "开发登录页面组件"},
            {"id": 4, "title": "联调与测试", "description": "前后端接口联调与功能测试"}
        ]
        return json.dumps(wbs, ensure_ascii=False)
    
    def _update_task_status_tool(self, task_id: int, status: str) -> str:
        """更新任务状态"""
        return f"任务 {task_id} 状态已更新为 {status}"
    
    def _work_on_task_tool(self, task_id: int, agent_name: str) -> str:
        """执行任务"""
        return f"{agent_name} 正在执行任务 {task_id}"
    
    def _review_api_draft_tool(self, api_draft: str) -> str:
        """评审API草案"""
        return "API草案已评审，发现需要确认错误处理格式"
    
    def _report_problem_tool(self, task_id: int, problem: str, agent_name: str) -> str:
        """报告问题"""
        return f"{agent_name} 报告任务 {task_id} 遇到问题: {problem}"
    
    def _create_api_draft_tool(self) -> str:
        """创建API草案"""
        return """
        【API草案】
        - 请求：POST /api/login, Body: { "email": "string", "password": "string", "rememberMe": "boolean" }
        - 响应：{ "code": 200, "message": "success", "data": { "token": "xxx", "userInfo": { ... } } }
        """
    
    def _assign_tasks_tool(self, assignments: str) -> str:
        """分配任务"""
        return f"任务已分配: {assignments}"
    
    def _make_decision_tool(self, decision_request_id: int, decision: int) -> str:
        """做出决策"""
        return f"已对决策请求 {decision_request_id} 做出选择: {decision}"
    
    def _check_progress_tool(self) -> str:
        """检查进度"""
        return "项目正在进行中"
    
    # 节点函数
    async def _planner_node(self, state: ProjectState):
        """规划专家节点"""
        if not state.tasks:
            # 生成WBS
            prompt = f"为项目目标 '{state.project_goal}' 生成详细的工作分解结构(WBS)"
            response = await self.planner_agent.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })
            
            # 解析生成的WBS并创建任务
            wbs_data = json.loads(self._generate_wbs_tool(state.project_goal))
            tasks = []
            for task_data in wbs_data:
                tasks.append(Task(
                    id=task_data["id"],
                    title=task_data["title"],
                    description=task_data["description"],
                    status=TaskStatus.TODO,
                    assigned_to=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ))
            
            state.tasks = tasks
            state.messages.append(HumanMessage(content=prompt))
            state.messages.append(AIMessage(content=f"已为项目生成{len(tasks)}个任务"))
        
        state.current_agent = AgentType.PROJECT_MANAGER
        return state
    
    async def _frontend_node(self, state: ProjectState):
        """前端开发节点"""
        # 查找分配给前端的任务
        frontend_tasks = [t for t in state.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.TODO]

        if frontend_tasks:
            task = frontend_tasks[0]

            # 将任务状态改为进行中
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.now()
            state.messages.append(AIMessage(content=f"前端开发开始执行任务: {task.title}"))

            # 检查是否有API草案需要评审
            if state.api_draft:
                prompt = f"请评审以下API草案并提出问题:\n{state.api_draft}"
                response = await self.frontend_agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })

                # 创建决策请求
                decision_request = DecisionRequest(
                    task_id=task.id,
                    from_agent="前端开发",
                    problem="关于登录失败的响应格式。草案中只定义了成功的响应。如果用户密码错误，后端应该返回什么？这会影响我的前端错误处理逻辑。",
                    options=[
                        "统一使用HTTP状态码200，所有业务错误通过响应体中的code字段区分",
                        "业务错误直接对应不同的HTTP状态码（如401未授权等）"
                    ]
                )
                state.decision_requests.append(decision_request)
                state.messages.append(AIMessage(content="前端开发需要确认API草案中的错误处理格式"))
            else:
                # 正常执行任务
                prompt = f"执行前端任务: {task.title} - {task.description}"
                response = await self.frontend_agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })
                task.status = TaskStatus.COMPLETED
                task.updated_at = datetime.now()
                state.messages.append(AIMessage(content=f"前端开发已完成任务: {task.title}"))
        else:
            # 检查是否有正在进行的任务需要处理决策
            frontend_in_progress = [t for t in state.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.IN_PROGRESS]
            if frontend_in_progress and state.api_draft:
                task = frontend_in_progress[0]
                prompt = f"请评审以下API草案并提出问题:\n{state.api_draft}"
                response = await self.frontend_agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })

                # 创建决策请求
                decision_request = DecisionRequest(
                    task_id=task.id,
                    from_agent="前端开发",
                    problem="关于登录失败的响应格式。草案中只定义了成功的响应。如果用户密码错误，后端应该返回什么？这会影响我的前端错误处理逻辑。",
                    options=[
                        "统一使用HTTP状态码200，所有业务错误通过响应体中的code字段区分",
                        "业务错误直接对应不同的HTTP状态码（如401未授权等）"
                    ]
                )
                state.decision_requests.append(decision_request)
                state.messages.append(AIMessage(content="前端开发需要确认API草案中的错误处理格式"))

        state.current_agent = AgentType.PROJECT_MANAGER
        return state
    
    async def _backend_node(self, state: ProjectState):
        """后端开发节点"""
        # 查找分配给后端的任务
        backend_tasks = [t for t in state.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.TODO]

        if backend_tasks:
            task = backend_tasks[0]

            # 将任务状态改为进行中
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.now()
            state.messages.append(AIMessage(content=f"后端开发开始执行任务: {task.title}"))

            if task.title == "后端API开发" and not state.api_draft:
                # 创建API草案
                prompt = "为登录功能创建API接口草案"
                response = await self.backend_agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })

                state.api_draft = self._create_api_draft_tool()

                # 创建决策请求
                decision_request = DecisionRequest(
                    task_id=task.id,
                    from_agent="后端开发",
                    problem="在开发登录接口时，我需要明确知道前端传递登录数据的格式，以及期望后端返回的响应体格式。没有这个约定，我无法继续编码。",
                    options=[
                        "请前端开发优先输出一份正式的接口文档",
                        "我可以先按照惯例定义一份草案，交由前端确认后再开发"
                    ]
                )
                state.decision_requests.append(decision_request)
                state.messages.append(AIMessage(content="后端开发需要确定接口格式约定"))
            else:
                # 正常执行任务
                prompt = f"执行后端任务: {task.title} - {task.description}"
                response = await self.backend_agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })
                task.status = TaskStatus.COMPLETED
                task.updated_at = datetime.now()
                state.messages.append(AIMessage(content=f"后端开发已完成任务: {task.title}"))
        else:
            # 检查是否有正在进行的任务需要继续
            backend_in_progress = [t for t in state.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.IN_PROGRESS]
            if backend_in_progress:
                task = backend_in_progress[0]
                if task.title == "后端API开发" and not state.api_draft:
                    # 创建API草案
                    prompt = "为登录功能创建API接口草案"
                    response = await self.backend_agent.ainvoke({
                        "messages": [HumanMessage(content=prompt)]
                    })

                    state.api_draft = self._create_api_draft_tool()

                    # 创建决策请求
                    decision_request = DecisionRequest(
                        task_id=task.id,
                        from_agent="后端开发",
                        problem="在开发登录接口时，我需要明确知道前端传递登录数据的格式，以及期望后端返回的响应体格式。没有这个约定，我无法继续编码。",
                        options=[
                            "请前端开发优先输出一份正式的接口文档",
                            "我可以先按照惯例定义一份草案，交由前端确认后再开发"
                        ]
                    )
                    state.decision_requests.append(decision_request)
                    state.messages.append(AIMessage(content="后端开发需要确定接口格式约定"))

        state.current_agent = AgentType.PROJECT_MANAGER
        return state
    
    async def _manager_node(self, state: ProjectState):
        """项目经理节点"""
        if not state.project_goal:
            # 等待用户输入项目目标 - 不改变current_agent，让路由处理
            return state

        # 检查是否有任务，如果没有则先让规划专家生成任务
        if not state.tasks:
            state.current_agent = AgentType.PLANNER
            return state

        # 检查是否需要分配任务
        unassigned_tasks = [t for t in state.tasks if t.assigned_to is None]
        if unassigned_tasks:
            prompt = f"请为以下任务分配负责人:\n" + "\n".join([
                f"{t.id}. {t.title} - {t.description}" for t in unassigned_tasks
            ])
            response = await self.manager_agent.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })

            # 简化的任务分配逻辑
            for task in unassigned_tasks:
                if "数据库" in task.title or "后端" in task.title:
                    task.assigned_to = "后端开发"
                elif "前端" in task.title:
                    task.assigned_to = "前端开发"
                else:
                    task.assigned_to = "后端开发"  # 默认分配给后端
                task.status = TaskStatus.TODO  # 先设为TODO，等待执行
                task.updated_at = datetime.now()

            state.messages.append(AIMessage(content="任务已分配完成，准备开始执行"))

            # 分配完任务后，立即开始执行后端任务
            state.current_agent = AgentType.BACKEND_DEV
            return state

        # 检查是否有待处理的决策请求
        pending_decisions = [dr for dr in state.decision_requests if dr.selected_option is None]
        if pending_decisions:
            # 保持PROJECT_MANAGER状态，让路由到human_input
            return state

        # 检查是否所有任务都已完成
        incomplete_tasks = [t for t in state.tasks if t.status != TaskStatus.COMPLETED]
        if not incomplete_tasks:
            state.completed = True
            state.messages.append(AIMessage(content="所有任务已完成！项目成功结束。"))
            return state

        # 决定下一步执行哪个开发人员
        frontend_tasks = [t for t in state.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.TODO]
        backend_tasks = [t for t in state.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.TODO]

        if backend_tasks:
            state.current_agent = AgentType.BACKEND_DEV
        elif frontend_tasks:
            state.current_agent = AgentType.FRONTEND_DEV
        else:
            # 如果没有TODO任务，检查是否有IN_PROGRESS任务需要继续
            frontend_in_progress = [t for t in state.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.IN_PROGRESS]
            backend_in_progress = [t for t in state.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.IN_PROGRESS]

            if backend_in_progress:
                state.current_agent = AgentType.BACKEND_DEV
            elif frontend_in_progress:
                state.current_agent = AgentType.FRONTEND_DEV
            else:
                # 真正没有任务了，保持manager状态
                pass

        return state
    
    async def _human_input_node(self, state: ProjectState):
        """人类输入节点"""
        # 这里处理需要人类输入的情况
        # 在实际应用中，这里会有UI交互
        return state
    
    # 路由函数
    def _route_after_manager(self, state: ProjectState) -> str:
        """经理节点后的路由"""
        if state.completed:
            return "end"

        if not state.project_goal:
            return "human_input"

        pending_decisions = [dr for dr in state.decision_requests if dr.selected_option is None]
        if pending_decisions:
            return "human_input"

        # 优先检查明确设置的下一个agent
        if hasattr(state, 'current_agent') and state.current_agent != AgentType.PROJECT_MANAGER:
            if state.current_agent == AgentType.PLANNER:
                return "planner"
            elif state.current_agent == AgentType.FRONTEND_DEV:
                return "frontend"
            elif state.current_agent == AgentType.BACKEND_DEV:
                return "backend"

        # 如果没有任务，去规划
        if not state.tasks:
            return "planner"

        # 如果有待分配的任务，继续在manager处理
        unassigned_tasks = [t for t in state.tasks if t.assigned_to is None]
        if unassigned_tasks:
            return "manager"

        # 优先执行后端任务
        backend_tasks = [t for t in state.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.TODO]
        if backend_tasks:
            return "backend"

        # 然后执行前端任务
        frontend_tasks = [t for t in state.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.TODO]
        if frontend_tasks:
            return "frontend"

        # 如果有进行中的任务，继续它们
        backend_in_progress = [t for t in state.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.IN_PROGRESS]
        if backend_in_progress:
            return "backend"

        frontend_in_progress = [t for t in state.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.IN_PROGRESS]
        if frontend_in_progress:
            return "frontend"

        # 默认结束或等待
        return "end"
    
    def _route_after_planner(self, state: ProjectState) -> str:
        """规划专家节点后的路由"""
        return "manager"
    
    def _route_after_developer(self, state: ProjectState) -> str:
        """开发人员节点后的路由"""
        return "manager"
    
    def _route_after_human(self, state: ProjectState) -> str:
        """人类输入节点后的路由"""
        return "manager"

class ProjectManagerSimulator:
    """项目管理模拟器"""
    
    def __init__(self):
        self.agents = ProjectManagementAgents()
        self.state = ProjectState(
            messages=[],
            project_goal="",
            tasks=[],
            decision_requests=[],
            current_agent=AgentType.PROJECT_MANAGER
        )
    
    def display_state(self):
        """显示当前状态"""
        print("\n" + "="*60)
        print("📋 AI项目管理实训系统 - 当前状态")
        print("="*60)
        
        print(f"\n🎯 项目目标: {self.state.project_goal or '未设置'}")
        
        print(f"\n👥 当前执行者: {self.state.current_agent.value}")
        
        print("\n📝 任务看板:")
        for status in TaskStatus:
            print(f"\n  {status.value}:")
            status_tasks = [t for t in self.state.tasks if t.status == status]
            if not status_tasks:
                print("    (无)")
            for task in status_tasks:
                assignee = task.assigned_to if task.assigned_to else "未分配"
                print(f"    {task.id}. {task.title} - [{assignee}]")
        
        print("\n🔴 待处理决策:")
        pending_decisions = [dr for dr in self.state.decision_requests if dr.selected_option is None]
        if not pending_decisions:
            print("  (无)")
        for i, decision in enumerate(pending_decisions):
            print(f"  {i+1}. 来自: {decision.from_agent}")
            print(f"     问题: {decision.problem}")
            print(f"     选项:")
            for j, option in enumerate(decision.options):
                print(f"       {j+1}. {option}")
    
    def display_messages(self):
        """显示消息历史"""
        print("\n💬 消息历史:")
        print("-" * 30)
        for msg in self.state.messages[-5:]:  # 显示最近5条消息
            if isinstance(msg, HumanMessage):
                print(f"👤: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"🤖: {msg.content}")
    
    async def run(self):
        """运行模拟器"""
        print("🚀 AI项目管理实训系统 (基于LangGraph)")
        print("="*40)

        # 获取项目目标
        self.state.project_goal = input("请输入项目目标: ").strip()
        self.state.messages.append(HumanMessage(content=f"项目目标: {self.state.project_goal}"))

        # 主循环
        while not self.state.completed:
            self.display_state()
            self.display_messages()

            # 检查是否需要人类决策
            pending_decisions = [dr for dr in self.state.decision_requests if dr.selected_option is None]
            if pending_decisions:
                print(f"\n🔴 需要您的决策 (共{len(pending_decisions)}个待处理):")
                for i, decision in enumerate(pending_decisions):
                    print(f"\n决策 {i+1}: 来自 {decision.from_agent}")
                    print(f"问题: {decision.problem}")
                    print("选项:")
                    for j, option in enumerate(decision.options):
                        print(f"  {j+1}. {option}")

                    try:
                        choice = int(input("请选择 (输入选项编号): "))
                        if 1 <= choice <= len(decision.options):
                            decision.selected_option = choice - 1
                            self.state.messages.append(
                                HumanMessage(content=f"项目经理决策: 选择方案 {choice}")
                            )
                            self.state.messages.append(
                                AIMessage(content=f"已执行决策: {decision.options[decision.selected_option]}")
                            )

                            # 特殊处理：如果是后端起草API的决策
                            if decision.from_agent == "后端开发" and decision.selected_option == 1:
                                self.state.api_draft = self.agents._create_api_draft_tool()
                                print(f"\n📄 后端开发已生成API草案:")
                                print(self.state.api_draft)
                        else:
                            print("无效选择")
                    except ValueError:
                        print("输入无效，请输入数字")

            # 执行下一步
            print("\n⏳ 执行下一步...")
            result = await self.agents.workflow.ainvoke(self.state)

            # 将字典结果转换回ProjectState对象
            if isinstance(result, dict):
                # 重建任务对象
                tasks = []
                for task_data in result.get('tasks', []):
                    if isinstance(task_data, dict):
                        tasks.append(Task(
                            id=task_data['id'],
                            title=task_data['title'],
                            description=task_data['description'],
                            status=TaskStatus(task_data['status']),
                            assigned_to=task_data['assigned_to'],
                            created_at=task_data['created_at'],
                            updated_at=task_data['updated_at']
                        ))
                    else:
                        tasks.append(task_data)

                # 重建决策请求对象
                decision_requests = []
                for dr_data in result.get('decision_requests', []):
                    if isinstance(dr_data, dict):
                        decision_requests.append(DecisionRequest(
                            task_id=dr_data['task_id'],
                            from_agent=dr_data['from_agent'],
                            problem=dr_data['problem'],
                            options=dr_data['options'],
                            selected_option=dr_data['selected_option']
                        ))
                    else:
                        decision_requests.append(dr_data)

                self.state = ProjectState(
                    messages=result.get('messages', []),
                    project_goal=result.get('project_goal', ''),
                    tasks=tasks,
                    decision_requests=decision_requests,
                    current_agent=AgentType(result.get('current_agent', AgentType.PROJECT_MANAGER)),
                    api_draft=result.get('api_draft', ''),
                    completed=result.get('completed', False),
                    current_task_id=result.get('current_task_id')
                )
            else:
                self.state = result

            # 短暂暂停以便观察
            await asyncio.sleep(1)

        # 显示最终报告
        self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终报告"""
        print("\n" + "="*60)
        print("📊 项目最终报告")
        print("="*60)
        
        completed_tasks = len([t for t in self.state.tasks if t.status == TaskStatus.COMPLETED])
        total_tasks = len(self.state.tasks)
        
        print(f"\n🎯 项目目标: {self.state.project_goal}")
        print(f"✅ 完成进度: {completed_tasks}/{total_tasks} 任务")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.state.decision_requests:
            print("\n🔍 关键决策点回顾:")
            for i, dr in enumerate(self.state.decision_requests):
                if dr.selected_option is not None:
                    print(f"  {i+1}. {dr.from_agent}: {dr.problem[:50]}...")
                    print(f"     决策: {dr.options[dr.selected_option]}")
        
        print("\n💡 经验学习:")
        print("  • 在任务分解时，可以增加一个'前后端接口约定'的独立任务")
        print("  • 确立统一的错误处理规范有助于提高开发效率")
        print("  • 智能体协作需要清晰的沟通协议和决策机制")
        print("  • 项目经理的关键作用在于协调和决策")

async def main():
    """主函数"""
    simulator = ProjectManagerSimulator()
    await simulator.run()

if __name__ == "__main__":
    asyncio.run(main())