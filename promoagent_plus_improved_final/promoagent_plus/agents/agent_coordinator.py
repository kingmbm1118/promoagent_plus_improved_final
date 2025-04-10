"""
Agent coordination system for ProMoAgent+
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from crewai import Agent, Task, Crew, Process

from promoagent_plus.utils.error_handling import ModelError, ErrorType, ErrorSeverity, safe_execute
from promoagent_plus.core.powl_utils import extract_powl_code, execute_powl_code

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enum for task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskResult:
    """Class to store task execution results"""
    
    def __init__(
        self, 
        task_id: str, 
        status: TaskStatus, 
        result: Any = None, 
        error: Optional[ModelError] = None
    ):
        """
        Initialize TaskResult
        
        Args:
            task_id: Unique identifier for the task
            status: Status of the task
            result: Result of the task execution
            error: Error that occurred during task execution
        """
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.attempts = 1
        self.feedback = []
    
    def add_feedback(self, feedback: str):
        """
        Add feedback for the task
        
        Args:
            feedback: Feedback message
        """
        self.feedback.append(feedback)
    
    def increment_attempts(self):
        """Increment the number of attempts"""
        self.attempts += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error.to_dict() if self.error else None,
            "attempts": self.attempts,
            "feedback": self.feedback
        }


class TaskManager:
    """
    Manages task creation, assignment, and tracking
    """
    
    def __init__(self):
        """Initialize TaskManager"""
        self.tasks = {}
        self.task_dependencies = {}
        self.task_results = {}
    
    def create_task(
        self, 
        task_id: str, 
        agent: Agent, 
        description: str, 
        expected_output: str,
        context: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Task:
        """
        Create a new task
        
        Args:
            task_id: Unique identifier for the task
            agent: Agent to assign the task to
            description: Task description
            expected_output: Expected output format
            context: Additional context for the task
            dependencies: List of task IDs that must be completed before this task
        
        Returns:
            Created Task object
        """
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context or {}
        )
        
        self.tasks[task_id] = task
        self.task_dependencies[task_id] = dependencies or []
        self.task_results[task_id] = TaskResult(task_id, TaskStatus.PENDING)
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID
        
        Args:
            task_id: Task ID
        
        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get task result by ID
        
        Args:
            task_id: Task ID
        
        Returns:
            TaskResult object or None if not found
        """
        return self.task_results.get(task_id)
    
    def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        result: Any = None, 
        error: Optional[ModelError] = None
    ):
        """
        Update task status
        
        Args:
            task_id: Task ID
            status: New status
            result: Task result
            error: Error that occurred
        """
        if task_id in self.task_results:
            task_result = self.task_results[task_id]
            task_result.status = status
            
            if result is not None:
                task_result.result = result
            
            if error is not None:
                task_result.error = error
    
    def get_ready_tasks(self) -> List[str]:
        """
        Get list of tasks that are ready to be executed
        
        Returns:
            List of task IDs
        """
        ready_tasks = []
        
        for task_id, dependencies in self.task_dependencies.items():
            # Skip tasks that are not pending
            if self.task_results[task_id].status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            all_dependencies_completed = True
            for dep_id in dependencies:
                if dep_id not in self.task_results or self.task_results[dep_id].status != TaskStatus.COMPLETED:
                    all_dependencies_completed = False
                    break
            
            if all_dependencies_completed:
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def get_task_dependency_results(self, task_id: str) -> Dict[str, Any]:
        """
        Get results of all dependencies for a task
        
        Args:
            task_id: Task ID
        
        Returns:
            Dictionary mapping dependency task IDs to their results
        """
        dependency_results = {}
        
        for dep_id in self.task_dependencies.get(task_id, []):
            if dep_id in self.task_results and self.task_results[dep_id].status == TaskStatus.COMPLETED:
                dependency_results[dep_id] = self.task_results[dep_id].result
        
        return dependency_results
    
    def all_tasks_completed(self) -> bool:
        """
        Check if all tasks are completed
        
        Returns:
            True if all tasks are completed, False otherwise
        """
        return all(
            result.status == TaskStatus.COMPLETED
            for result in self.task_results.values()
        )
    
    def get_failed_tasks(self) -> List[str]:
        """
        Get list of failed tasks
        
        Returns:
            List of failed task IDs
        """
        return [
            task_id
            for task_id, result in self.task_results.items()
            if result.status == TaskStatus.FAILED
        ]


class CollaborationContext:
    """
    Shared context for agent collaboration
    """
    
    def __init__(self):
        """Initialize CollaborationContext"""
        self.shared_data = {}
        self.history = []
    
    def set(self, key: str, value: Any):
        """
        Set a value in the shared context
        
        Args:
            key: Data key
            value: Data value
        """
        self.shared_data[key] = value
        self.history.append({
            "action": "set",
            "key": key,
            "value_type": type(value).__name__
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared context
        
        Args:
            key: Data key
            default: Default value if key not found
        
        Returns:
            Value associated with the key or default
        """
        value = self.shared_data.get(key, default)
        self.history.append({
            "action": "get",
            "key": key,
            "found": key in self.shared_data
        })
        return value
    
    def update(self, data: Dict[str, Any]):
        """
        Update multiple values in the shared context
        
        Args:
            data: Dictionary of key-value pairs to update
        """
        self.shared_data.update(data)
        self.history.append({
            "action": "update",
            "keys": list(data.keys())
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of context operations
        
        Returns:
            List of history entries
        """
        return self.history
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all shared data
        
        Returns:
            Dictionary of all shared data
        """
        return self.shared_data.copy()


class MasterAgent:
    """
    Master agent that coordinates specialized agents
    """
    
    def __init__(
        self, 
        specialized_agents: Dict[str, Agent],
        max_retries: int = 3,
        monitoring_enabled: bool = True
    ):
        """
        Initialize MasterAgent
        
        Args:
            specialized_agents: Dictionary of specialized agents
            max_retries: Maximum number of retries for failed tasks
            monitoring_enabled: Whether to enable monitoring
        """
        self.specialized_agents = specialized_agents
        self.max_retries = max_retries
        self.task_manager = TaskManager()
        self.context = CollaborationContext()
        self.monitoring_enabled = monitoring_enabled
        
        if monitoring_enabled:
            from promoagent_plus.utils.monitoring import AgentMonitor
            self.monitor = AgentMonitor()
        else:
            self.monitor = None
    
    def create_workflow(
        self, 
        workflow_definition: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Create a workflow of tasks
        
        Args:
            workflow_definition: List of task definitions with:
                - id: Task ID
                - agent: Agent ID
                - description: Task description
                - expected_output: Expected output format
                - context: Additional context (optional)
                - dependencies: List of dependency task IDs (optional)
        
        Returns:
            List of created task IDs
        """
        task_ids = []
        
        for task_def in workflow_definition:
            task_id = task_def["id"]
            agent_id = task_def["agent"]
            
            if agent_id not in self.specialized_agents:
                raise ValueError(f"Unknown agent ID: {agent_id}")
            
            task = self.task_manager.create_task(
                task_id=task_id,
                agent=self.specialized_agents[agent_id],
                description=task_def["description"],
                expected_output=task_def["expected_output"],
                context=task_def.get("context"),
                dependencies=task_def.get("dependencies")
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def execute_workflow(self) -> Dict[str, Any]:
        """
        Execute the workflow of tasks
        
        Returns:
            Dictionary of task results
        """
        while not self.task_manager.all_tasks_completed():
            # Get tasks that are ready to be executed
            ready_tasks = self.task_manager.get_ready_tasks()
            
            if not ready_tasks:
                # Check if there are failed tasks that can be retried
                failed_tasks = self.task_manager.get_failed_tasks()
                
                if not failed_tasks:
                    # No ready tasks and no failed tasks to retry
                    # This could indicate a deadlock or all tasks are in progress
                    logger.warning("No ready tasks and no failed tasks to retry")
                    break
                
                # Try to retry failed tasks
                for task_id in failed_tasks:
                    task_result = self.task_manager.get_task_result(task_id)
                    
                    if task_result.attempts < self.max_retries:
                        logger.info(f"Retrying failed task: {task_id} (Attempt {task_result.attempts + 1}/{self.max_retries})")
                        task_result.increment_attempts()
                        self.task_manager.update_task_status(task_id, TaskStatus.RETRYING)
                        ready_tasks.append(task_id)
                    else:
                        logger.error(f"Task {task_id} failed after {self.max_retries} attempts")
            
            # Execute ready tasks
            for task_id in ready_tasks:
                self._execute_task(task_id)
        
        # Return final results
        return {
            task_id: result.to_dict()
            for task_id, result in self.task_manager.task_results.items()
        }
    
    def _execute_task(self, task_id: str):
        """
        Execute a single task
        
        Args:
            task_id: Task ID to execute
        """
        task = self.task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return
        
        # Update task status
        self.task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)
        
        # Get dependency results
        dependency_results = self.task_manager.get_task_dependency_results(task_id)
        
        # Update task context with dependency results
        task_context = task.context or {}
        task_context.update({
            "dependency_results": dependency_results,
            "shared_context": self.context.get_all()
        })
        
        # Create a crew with just this agent
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        
        # Execute the task
        logger.info(f"Executing task: {task_id}")
        
        try:
            # Execute with error handling
            success, result, error = safe_execute(
                crew.kickoff,
                error_context={"task_id": task_id, "agent": task.agent.role}
            )
            
            if success:
                # Task completed successfully
                self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result)
                logger.info(f"Task completed successfully: {task_id}")
                
                # Update shared context with task result
                self.context.set(f"task_result_{task_id}", result)
            else:
                # Task failed
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, None, error)
                logger.error(f"Task failed: {task_id} - {str(error)}")
                
                # Try to recover if possible
                if error and task.agent.role == "POWL Modeler" and "powl_code" in task_context:
                    # Attempt to fix POWL code
                    logger.info(f"Attempting to recover from error in task: {task_id}")
                    self._attempt_recovery(task_id, error, task_context)
        
        except Exception as e:
            # Unexpected error
            logger.exception(f"Unexpected error executing task {task_id}: {str(e)}")
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED)
    
    def _attempt_recovery(self, task_id: str, error: ModelError, context: Dict[str, Any]):
        """
        Attempt to recover from a task error
        
        Args:
            task_id: Failed task ID
            error: Error that occurred
            context: Task context
        """
        # This is a simplified recovery attempt focused on POWL code errors
        if error.error_type in [ErrorType.EXECUTION, ErrorType.VALIDATION] and "powl_code" in context:
            powl_code = context["powl_code"]
            
            # Create a recovery task for the Model Reviewer agent
            recovery_task_id = f"{task_id}_recovery"
            
            recovery_description = f"""
            The following POWL code has an error: {str(error)}
            
            ```python
            {powl_code}
            ```
            
            Please analyze the code and fix the error. The specific issue is: {str(error.original_exception) if error.original_exception else str(error)}
            
            Provide the corrected POWL code.
            """
            
            recovery_task = self.task_manager.create_task(
                task_id=recovery_task_id,
                agent=self.specialized_agents["model_reviewer"],
                description=recovery_description,
                expected_output="Corrected POWL code",
                context={"original_error": str(error), "powl_code": powl_code}
            )
            
            # Execute recovery task
            logger.info(f"Executing recovery task: {recovery_task_id}")
            
            crew = Crew(
                agents=[self.specialized_agents["model_reviewer"]],
                tasks=[recovery_task],
                verbose=True,
                process=Process.sequential
            )
            
            try:
                recovery_result = crew.kickoff()
                
                # Extract POWL code from recovery result
                try:
                    fixed_code = extract_powl_code(recovery_result)
                    
                    # Validate the fixed code
                    execute_powl_code(fixed_code)
                    
                    # Update the original task with the fixed code
                    self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED, fixed_code)
                    logger.info(f"Recovery successful for task: {task_id}")
                    
                    # Update shared context
                    self.context.set(f"task_result_{task_id}", fixed_code)
                    self.context.set(f"recovery_{task_id}", "successful")
                    
                except Exception as e:
                    logger.error(f"Recovery validation failed: {str(e)}")
                    self.context.set(f"recovery_{task_id}", "failed")
            
            except Exception as e:
                logger.error(f"Recovery task failed: {str(e)}")
                self.context.set(f"recovery_{task_id}", "failed")
    
    def generate_model_from_text(self, description: str) -> Dict[str, Any]:
        """
        Generate process model from textual description
        
        Args:
            description: Process description text
        
        Returns:
            Dictionary with generation results
        """
        # Define the workflow for text-to-model generation
        workflow = [
            {
                "id": "analyze_text",
                "agent": "process_analyzer",
                "description": f"Analyze the following process description and identify activities, actors, and control flow:\n\n{description}",
                "expected_output": "JSON with identified process components",
                "context": {"description": description}
            },
            {
                "id": "create_powl_model",
                "agent": "powl_modeler",
                "description": "Create a POWL model based on the process analysis",
                "expected_output": "POWL model code",
                "dependencies": ["analyze_text"]
            },
            {
                "id": "review_model",
                "agent": "model_reviewer",
                "description": "Review the POWL model for correctness and completeness",
                "expected_output": "Review feedback with suggestions for improvement",
                "dependencies": ["create_powl_model"]
            },
            {
                "id": "improve_model",
                "agent": "powl_modeler",
                "description": "Improve the POWL model based on the review feedback",
                "expected_output": "Improved POWL model code",
                "dependencies": ["review_model", "create_powl_model"]
            },
            {
                "id": "translate_to_bpmn",
                "agent": "model_translator",
                "description": "Translate the improved POWL model to BPMN",
                "expected_output": "Confirmation of successful translation",
                "dependencies": ["improve_model"]
            },
            {
                "id": "translate_to_petri_net",
                "agent": "model_translator",
                "description": "Translate the improved POWL model to Petri net",
                "expected_output": "Confirmation of successful translation",
                "dependencies": ["improve_model"]
            }
        ]
        
        # Create and execute the workflow
        self.create_workflow(workflow)
        results = self.execute_workflow()
        
        # Extract the final model
        final_model = None
        if "improve_model" in results and results["improve_model"]["status"] == "completed":
            final_model = results["improve_model"]["result"]
        elif "create_powl_model" in results and results["create_powl_model"]["status"] == "completed":
            final_model = results["create_powl_model"]["result"]
        
        return {
            "powl_code": final_model,
            "workflow_results": results,
            "context_history": self.context.get_history()
        }
    
    def generate_model_with_feedback(
        self, 
        description: str, 
        feedback_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Generate process model with iterative feedback
        
        Args:
            description: Process description text
            feedback_iterations: Number of feedback iterations
        
        Returns:
            Dictionary with generation results
        """
        # Initial model generation
        initial_result = self.generate_model_from_text(description)
        
        # Extract the initial model
        current_model = None
        if "improve_model" in initial_result["workflow_results"] and initial_result["workflow_results"]["improve_model"]["status"] == "completed":
            current_model = initial_result["workflow_results"]["improve_model"]["result"]
        elif "create_powl_model" in initial_result["workflow_results"] and initial_result["workflow_results"]["create_powl_model"]["status"] == "completed":
            current_model = initial_result["workflow_results"]["create_powl_model"]["result"]
        
        if not current_model:
            return initial_result
        
        # Iterative feedback and improvement
        feedback_results = []
        
        for i in range(feedback_iterations):
            # Reset task manager for new iteration
            self.task_manager = TaskManager()
            
            # Define feedback workflow
            feedback_workflow = [
                {
                    "id": f"review_iteration_{i+1}",
                    "agent": "model_reviewer",
                    "description": f"Review the current POWL model for iteration {i+1}:\n\n```python\n{current_model}\n```\n\nProvide detailed feedback for further improvement.",
                    "expected_output": "Detailed review feedback",
                    "context": {"current_model": current_model, "iteration": i+1}
                },
                {
                    "id": f"improve_iteration_{i+1}",
                    "agent": "powl_modeler",
                    "description": "Improve the POWL model based on the review feedback",
                    "expected_output": "Improved POWL model code",
                    "dependencies": [f"review_iteration_{i+1}"],
                    "context": {"current_model": current_model, "iteration": i+1}
                }
            ]
            
            # Create and execute feedback workflow
            self.create_workflow(feedback_workflow)
            iteration_results = self.execute_workflow()
            
            # Update current model
            if f"improve_iteration_{i+1}" in iteration_results and iteration_results[f"improve_iteration_{i+1}"]["status"] == "completed":
                current_model = iteration_results[f"improve_iteration_{i+1}"]["result"]
            
            feedback_results.append({
                "iteration": i+1,
                "results": iteration_results
            })
        
        # Combine results
        return {
            "initial_generation": initial_result,
            "feedback_iterations": feedback_results,
            "final_powl_code": current_model
        }
