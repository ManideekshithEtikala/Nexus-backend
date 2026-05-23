# backend/core/orchestration.py
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.config import settings
from core.agents import run_subagent

class SubTask(BaseModel):
    id: int = Field(description="Unique incremental ID of the sub-task")
    description: str = Field(description="Detailed instruction of what this sub-task must accomplish")
    assigned_agent: str = Field(description="Must be one of: 'coder', 'shell', 'git_agent', 'researcher'")
    dependencies: List[int] = Field(default=[], description="IDs of sub-tasks that must be completed before this one starts")

class TaskPlan(BaseModel):
    plan_checklist: List[SubTask] = Field(description="Ordered list of independent or dependent sub-tasks to achieve the user goal")

class TaskSupervisor:
    """
    Supervisor Agent that decomposes user requests into a structured DAG checklist,
    dispatches tasks to worker subagents, and compiles the final aggregated response.
    """
    def __init__(self, stream_queue: asyncio.Queue = None, hybrid_context: str = ""):
        self.stream_queue = stream_queue
        self.hybrid_context = hybrid_context
        # Initialize Groq LLM for planning
        self.llm = ChatGroq(
            model=settings.API_KEY and "qwen/qwen3-32b" or "llama-3.3-70b-versatile",
            api_key=settings.API_KEY,
            temperature=0.2
        )
        self.parser = PydanticOutputParser(pydantic_object=TaskPlan)

    async def generate_plan(self, user_query: str, history_context: str = "") -> TaskPlan:
        """Analyze user query and generate a structured checklist of tasks."""
        if self.stream_queue:
            await self.stream_queue.put({
                "type": "reasoning",
                "content": "\n\n[Supervisor] Analyzing request and drafting structured execution plan...\n"
            })

        planning_prompt = ChatPromptTemplate.from_template(
            "You are the Nexus Task Supervisor. Analyze the user request and decompose it into an ordered list "
            "of modular sub-tasks that can be executed by specialized worker agents (coder, shell, git_agent, researcher).\n\n"
            "Guidelines:\n"
            "1. Coder: for viewing directory structures, reading/writing/searching/modifying code files.\n"
            "2. Shell: for executing bash commands, running python scripts, check OS system tools.\n"
            "3. Git_agent: for staging changes, running git diff, committing, or checking git logs.\n"
            "4. Researcher: for crawling web pages, checking PyPI packages, or gathering external docs.\n\n"
            "{hybrid_context}\n\n"
            "User Request: {query}\n"
            "History Context: {context}\n\n"
            "{format_instructions}\n"
        )

        prompt_val = planning_prompt.format_prompt(
            query=user_query,
            context=history_context,
            hybrid_context=self.hybrid_context,
            format_instructions=self.parser.get_format_instructions()
        )

        try:
            response = await self.llm.ainvoke(prompt_val.to_messages())
            plan = self.parser.parse(response.content)
            
            if self.stream_queue:
                stream_text = "\n[Supervisor] Generated Structured Plan Checklist:\n"
                for task in plan.plan_checklist:
                    deps = f" (depends on {task.dependencies})" if task.dependencies else ""
                    stream_text += f"  - [{task.id}] {task.description} -> [{task.assigned_agent.upper()}]{deps}\n"
                await self.stream_queue.put({"type": "reasoning", "content": stream_text + "\n"})
                
            return plan
        except Exception as e:
            print(f"[Supervisor] Planning failed, falling back to sequential plan: {e}")
            # Dynamic fallback planning if parse fails
            return TaskPlan(plan_checklist=[
                SubTask(id=1, description=f"Execute task: {user_query}", assigned_agent="coder")
            ])

    async def execute_plan(self, plan: TaskPlan) -> str:
        """Iterate and execute each sub-task in the checklist, handling dependencies dynamically."""
        task_results = {}
        
        for task in plan.plan_checklist:
            if self.stream_queue:
                await self.stream_queue.put({
                    "type": "reasoning",
                    "content": f"[Supervisor] Dispatched Task {task.id}: '{task.description}' to {task.assigned_agent.upper()} worker...\n"
                })

            # Wait for dependencies
            for dep_id in task.dependencies:
                while dep_id not in task_results:
                    await asyncio.sleep(0.5)

            # Compile context from previous tasks
            context_payload = f"Task Description: {task.description}\n"
            if task.dependencies:
                context_payload += "\n=== PREVIOUS TASK INPUTS & OBSERVATIONS ===\n"
                for dep_id in task.dependencies:
                    context_payload += f"Task {dep_id} Result: {task_results[dep_id]}\n"

            # Execute subagent
            worker_result = await run_subagent(task.assigned_agent, context_payload, self.stream_queue)
            task_results[task.id] = worker_result

            if self.stream_queue:
                await self.stream_queue.put({
                    "type": "reasoning",
                    "content": f"[Supervisor] Task {task.id} Completed by {task.assigned_agent.upper()}!\n"
                })

        # Aggregated final compilation prompt
        compilation_prompt = ""
        if self.hybrid_context:
            compilation_prompt += f"{self.hybrid_context}\n\n"
            
        compilation_prompt += (
            "You are the Nexus Task Supervisor. Compile a friendly, highly polished, and professional final answer "
            "answering the user's original request directly based on the outcomes of all executed tasks in this plan. "
            "Do NOT include any raw transaction logs, database details, system schemas, or agent execution steps.\n\n"
            "Here is the execution history of the tasks performed:\n\n"
        )
        for t_id, t_res in task_results.items():
            compilation_prompt += f"- Task {t_id} Outcome: {t_res}\n"
            
        compilation_prompt += (
            "\nFormulate your final response now. Do NOT output any reasoning, decision protocol thoughts, or "
            "meta-chat about missing data. Wrap your direct answer strictly inside <final_answer></final_answer> tags."
        )
        
        response = await self.llm.ainvoke(compilation_prompt)
        return response.content
