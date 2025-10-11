#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

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

class SimpleProjectManager:
    """简化的项目管理模拟器"""

    def __init__(self):
        # 从环境变量读取配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key == "your-openai-api-key-here":
            print("警告：未设置有效的OpenAI API密钥，使用演示模式")
            self.api_key = "sk-test-key-for-demo"

        self.api_base = os.getenv("OPENAI_API_BASE")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        try:
            self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except ValueError:
            self.temperature = 0.7

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base
        )

        # 项目状态
        self.project_goal = ""
        self.tasks: List[Task] = []
        self.decision_requests: List[DecisionRequest] = []
        self.messages: List[Any] = []
        self.api_draft = ""
        self.completed = False

    def display_state(self):
        """显示当前状态"""
        print("\n" + "="*60)
        print("AI项目管理实训系统 - 当前状态")
        print("="*60)

        print(f"\n项目目标: {self.project_goal or '未设置'}")

        print("\n任务看板:")
        for status in TaskStatus:
            print(f"\n  {status.value}:")
            status_tasks = [t for t in self.tasks if t.status == status]
            if not status_tasks:
                print("    (无)")
            for task in status_tasks:
                assignee = task.assigned_to if task.assigned_to else "未分配"
                print(f"    {task.id}. {task.title} - [{assignee}]")

        print("\n待处理决策:")
        pending_decisions = [dr for dr in self.decision_requests if dr.selected_option is None]
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
        print("\n消息历史:")
        print("-" * 30)
        for msg in self.messages[-5:]:  # 显示最近5条消息
            if isinstance(msg, HumanMessage):
                print(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"系统: {msg.content}")
            else:
                print(f"系统: {msg}")

    async def generate_wbs(self, project_goal: str) -> List[Task]:
        """生成工作分解结构"""
        self.messages.append(HumanMessage(content=f"为项目目标 '{project_goal}' 生成工作分解结构"))

        # 简化：使用固定模板
        wbs_data = [
            {"id": 1, "title": "数据库设计", "description": "创建用户表，设计字段"},
            {"id": 2, "title": "后端API开发", "description": "开发登录接口和登出接口"},
            {"id": 3, "title": "前端页面开发", "description": "开发登录页面组件"},
            {"id": 4, "title": "联调与测试", "description": "前后端接口联调与功能测试"}
        ]

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

        self.messages.append(AIMessage(content=f"已为项目生成{len(tasks)}个任务"))
        return tasks

    def assign_tasks(self):
        """分配任务"""
        unassigned_tasks = [t for t in self.tasks if t.assigned_to is None]

        for task in unassigned_tasks:
            if "数据库" in task.title or "后端" in task.title:
                task.assigned_to = "后端开发"
            elif "前端" in task.title:
                task.assigned_to = "前端开发"
            else:
                task.assigned_to = "后端开发"
            task.updated_at = datetime.now()

        self.messages.append(AIMessage(content="任务已分配完成"))

    async def execute_backend_task(self, task: Task):
        """执行后端任务"""
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        self.messages.append(AIMessage(content=f"后端开发开始执行任务: {task.title}"))

        # 模拟API开发任务
        if task.title == "后端API开发" and not self.api_draft:
            self.api_draft = """
【API草案】
- 请求：POST /api/login, Body: { "email": "string", "password": "string", "rememberMe": "boolean" }
- 响应：{ "code": 200, "message": "success", "data": { "token": "xxx", "userInfo": { ... } } }
"""

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
            self.decision_requests.append(decision_request)
            self.messages.append(AIMessage(content="后端开发需要确定接口格式约定"))
            return True  # 需要决策

        # 其他任务直接完成
        task.status = TaskStatus.COMPLETED
        task.updated_at = datetime.now()
        self.messages.append(AIMessage(content=f"后端开发已完成任务: {task.title}"))
        return False  # 不需要决策

    async def execute_frontend_task(self, task: Task):
        """执行前端任务"""
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        self.messages.append(AIMessage(content=f"前端开发开始执行任务: {task.title}"))

        # 检查是否有API草案需要评审
        if self.api_draft:
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
            self.decision_requests.append(decision_request)
            self.messages.append(AIMessage(content="前端开发需要确认API草案中的错误处理格式"))
            return True  # 需要决策
        else:
            # 直接完成任务
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.now()
            self.messages.append(AIMessage(content=f"前端开发已完成任务: {task.title}"))
            return False  # 不需要决策

    async def handle_decision(self, decision: DecisionRequest, choice: int):
        """处理用户决策"""
        decision.selected_option = choice - 1
        selected_option = decision.options[decision.selected_option]

        self.messages.append(HumanMessage(content=f"项目经理决策: 选择方案 {choice}"))
        self.messages.append(AIMessage(content=f"已执行决策: {selected_option}"))

        # 特殊处理：如果是后端起草API的决策
        if decision.from_agent == "后端开发" and decision.selected_option == 1:
            print(f"\n后端开发已生成API草案:")
            print(self.api_draft)

        # 解除任务阻塞
        for task in self.tasks:
            if task.id == decision.task_id:
                if task.status == TaskStatus.IN_PROGRESS:
                    task.status = TaskStatus.COMPLETED
                    task.updated_at = datetime.now()
                    self.messages.append(AIMessage(content=f"决策已执行，任务继续推进"))
                break

    def check_completion(self):
        """检查项目是否完成"""
        incomplete_tasks = [t for t in self.tasks if t.status != TaskStatus.COMPLETED]
        if not incomplete_tasks:
            self.completed = True
            self.messages.append(AIMessage(content="所有任务已完成！项目成功结束。"))
            return True
        return False

    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "="*60)
        print("项目最终报告")
        print("="*60)

        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        total_tasks = len(self.tasks)

        print(f"\n项目目标: {self.project_goal}")
        print(f"完成进度: {completed_tasks}/{total_tasks} 任务")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.decision_requests:
            print("\n关键决策点回顾:")
            for i, dr in enumerate(self.decision_requests):
                if dr.selected_option is not None:
                    print(f"  {i+1}. {dr.from_agent}: {dr.problem[:50]}...")
                    print(f"     决策: {dr.options[dr.selected_option]}")

        print("\n经验学习:")
        print("  • 在任务分解时，可以增加一个'前后端接口约定'的独立任务")
        print("  • 确立统一的错误处理规范有助于提高开发效率")
        print("  • 智能体协作需要清晰的沟通协议和决策机制")
        print("  • 项目经理的关键作用在于协调和决策")

    async def run(self):
        """运行模拟器"""
        print("AI项目管理实训系统 (简化版)")
        print("="*40)

        # 获取项目目标
        self.project_goal = input("请输入项目目标: ").strip()

        # 主循环
        while not self.completed:
            self.display_state()
            self.display_messages()

            # 检查是否需要人类决策
            pending_decisions = [dr for dr in self.decision_requests if dr.selected_option is None]
            if pending_decisions:
                print(f"\n需要您的决策 (共{len(pending_decisions)}个待处理):")
                for i, decision in enumerate(pending_decisions):
                    print(f"\n决策 {i+1}: 来自 {decision.from_agent}")
                    print(f"问题: {decision.problem}")
                    print("选项:")
                    for j, option in enumerate(decision.options):
                        print(f"  {j+1}. {option}")

                    try:
                        choice = int(input("请选择 (输入选项编号): "))
                        if 1 <= choice <= len(decision.options):
                            await self.handle_decision(decision, choice)
                        else:
                            print("无效选择")
                    except ValueError:
                        print("输入无效，请输入数字")

                continue  # 处理完决策后继续循环

            # 执行下一步
            print("\n执行下一步...")

            # 1. 如果没有任务，先生成任务
            if not self.tasks:
                self.tasks = await self.generate_wbs(self.project_goal)
                await asyncio.sleep(1)
                continue

            # 2. 如果有未分配的任务，先分配
            unassigned_tasks = [t for t in self.tasks if t.assigned_to is None]
            if unassigned_tasks:
                self.assign_tasks()
                await asyncio.sleep(1)
                continue

            # 3. 执行后端任务
            backend_tasks = [t for t in self.tasks if t.assigned_to == "后端开发" and t.status == TaskStatus.TODO]
            if backend_tasks:
                task = backend_tasks[0]
                needs_decision = await self.execute_backend_task(task)
                await asyncio.sleep(1)
                if needs_decision:
                    continue  # 需要决策，等待用户输入
                continue

            # 4. 执行前端任务
            frontend_tasks = [t for t in self.tasks if t.assigned_to == "前端开发" and t.status == TaskStatus.TODO]
            if frontend_tasks:
                task = frontend_tasks[0]
                needs_decision = await self.execute_frontend_task(task)
                await asyncio.sleep(1)
                if needs_decision:
                    continue  # 需要决策，等待用户输入
                continue

            # 5. 检查是否完成
            if self.check_completion():
                break

            # 如果没有更多任务可执行，等待一下
            print("等待中...")
            await asyncio.sleep(2)

        # 显示最终报告
        self.generate_final_report()

async def main():
    """主函数"""
    simulator = SimpleProjectManager()
    await simulator.run()

if __name__ == "__main__":
    asyncio.run(main())