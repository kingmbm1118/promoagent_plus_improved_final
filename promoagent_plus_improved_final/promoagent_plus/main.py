"""
Updated main module for ProMoAgent+ with enhanced error handling and agent collaboration
"""

from typing import Optional, Dict, Any
from crewai import Crew, Process, Agent, Task

from promoagent_plus.utils.constants import AIProviders, AI_MODEL_DEFAULTS, DEFAULT_AI_PROVIDER
from promoagent_plus.agents.agent_factory import create_agents
from promoagent_plus.agents.agent_coordinator import MasterAgent
from promoagent_plus.utils.error_handling import safe_execute, error_handler
from promoagent_plus.models.result import ProcessModelResult
from promoagent_plus.utils.converters import (
    convert_event_log_to_powl,
    convert_petri_net_to_powl,
    convert_bpmn_to_powl
)
from promoagent_plus.core.powl_utils import powl_to_code, execute_powl_code
from promoagent_plus.utils.enhanced_validation import validate_and_fix_powl_model

class ProMoAgentPlus:
    """
    Main class for the enhanced ProMoAgent+ implementation with improved error handling
    and agent collaboration
    """
    
    def __init__(
        self, 
        api_key: str, 
        ai_provider: str = DEFAULT_AI_PROVIDER, 
        model_name: Optional[str] = None,
        monitoring_enabled: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize ProMoAgent+
        
        Args:
            api_key: API key for the LLM provider
            ai_provider: AI provider to use (from AIProviders enum)
            model_name: Optional specific model name to use (defaults to provider default)
            monitoring_enabled: Whether to enable agent monitoring
            max_retries: Maximum number of retries for failed operations
        """
        self.api_key = api_key
        self.ai_provider = ai_provider
        self.model_name = model_name or AI_MODEL_DEFAULTS.get(ai_provider)
        self.max_retries = max_retries
        self.monitoring_enabled = monitoring_enabled
        
        # Create specialized agents
        self.agents = create_agents(api_key, ai_provider, self.model_name)
        
        # Create master agent for coordination
        self.master_agent = MasterAgent(
            specialized_agents={
                "process_analyzer": self.agents["process_analyzer"],
                "powl_modeler": self.agents["powl_modeler"],
                "model_reviewer": self.agents["model_reviewer"],
                "model_translator": self.agents["model_translator"]
            },
            max_retries=self.max_retries,
            monitoring_enabled=self.monitoring_enabled
        )
        
        # Initialize results storage
        self.results = {}
    
    def generate_model_from_text(self, description: str) -> ProcessModelResult:
        """
        Generate process model from textual description using multi-agent collaboration
        
        Args:
            description: Process description text
            
        Returns:
            ProcessModelResult object
        """
        # Use the master agent to coordinate the model generation
        generation_results = self.master_agent.generate_model_from_text(description)
        
        # Extract the final POWL code
        powl_code = generation_results.get("powl_code")
        
        # If no valid POWL code was generated, try to recover
        if not powl_code:
            # Check if we have any partial results
            workflow_results = generation_results.get("workflow_results", {})
            for task_id in ["improve_model", "create_powl_model"]:
                if task_id in workflow_results and workflow_results[task_id]["status"] == "completed":
                    partial_code = workflow_results[task_id]["result"]
                    
                    # Validate and fix the partial code
                    validation_result = validate_and_fix_powl_model(partial_code)
                    if validation_result["status"] in ["success", "warning"]:
                        powl_code = validation_result["fixed_code"]
                        break
        
        # Store results for reference
        self.results = generation_results
        
        # Create result object
        return ProcessModelResult(
            powl_code=powl_code,
            api_key=self.api_key,
            ai_provider=self.ai_provider,
            model_name=self.model_name,
            original_description=description,
            analysis=str(generation_results.get("workflow_results", {}).get("analyze_text", {}).get("result", "")),
            review=str(generation_results.get("workflow_results", {}).get("review_model", {}).get("result", ""))
        )
    
    def generate_model_with_feedback(
        self, 
        description: str, 
        feedback_iterations: int = 2
    ) -> ProcessModelResult:
        """
        Generate process model with iterative feedback
        
        Args:
            description: Process description text
            feedback_iterations: Number of feedback iterations
            
        Returns:
            ProcessModelResult object
        """
        # Use the master agent to coordinate the model generation with feedback
        generation_results = self.master_agent.generate_model_with_feedback(
            description, 
            feedback_iterations
        )
        
        # Extract the final POWL code
        powl_code = generation_results.get("final_powl_code")
        
        # If no valid POWL code was generated, try to recover
        if not powl_code:
            # Check initial generation results
            initial_generation = generation_results.get("initial_generation", {})
            workflow_results = initial_generation.get("workflow_results", {})
            
            for task_id in ["improve_model", "create_powl_model"]:
                if task_id in workflow_results and workflow_results[task_id]["status"] == "completed":
                    partial_code = workflow_results[task_id]["result"]
                    
                    # Validate and fix the partial code
                    validation_result = validate_and_fix_powl_model(partial_code)
                    if validation_result["status"] in ["success", "warning"]:
                        powl_code = validation_result["fixed_code"]
                        break
            
            # Check feedback iteration results if still no valid code
            if not powl_code:
                feedback_iterations = generation_results.get("feedback_iterations", [])
                for iteration in reversed(feedback_iterations):  # Start from the latest iteration
                    iteration_results = iteration.get("results", {})
                    improve_task_id = f"improve_iteration_{iteration.get('iteration')}"
                    
                    if improve_task_id in iteration_results and iteration_results[improve_task_id]["status"] == "completed":
                        partial_code = iteration_results[improve_task_id]["result"]
                        
                        # Validate and fix the partial code
                        validation_result = validate_and_fix_powl_model(partial_code)
                        if validation_result["status"] in ["success", "warning"]:
                            powl_code = validation_result["fixed_code"]
                            break
        
        # Store results for reference
        self.results = generation_results
        
        # Create result object
        return ProcessModelResult(
            powl_code=powl_code,
            api_key=self.api_key,
            ai_provider=self.ai_provider,
            model_name=self.model_name,
            original_description=description,
            analysis=str(generation_results.get("initial_generation", {}).get("workflow_results", {}).get("analyze_text", {}).get("result", "")),
            review=str(generation_results.get("initial_generation", {}).get("workflow_results", {}).get("review_model", {}).get("result", ""))
        )
    
    def generate_model_from_event_log(self, event_log_path: str) -> ProcessModelResult:
        """
        Generate process model from event log with enhanced error handling
        
        Args:
            event_log_path: Path to the event log file
            
        Returns:
            ProcessModelResult object
        """
        # Convert event log to POWL with error handling
        success, powl_model, error = safe_execute(
            convert_event_log_to_powl,
            error_context={"operation": "convert_event_log_to_powl"},
            event_log_path=event_log_path
        )
        
        if not success:
            raise Exception(f"Failed to convert event log to POWL: {str(error)}")
        
        # Convert POWL to code with error handling
        success, powl_code, error = safe_execute(
            powl_to_code,
            error_context={"operation": "powl_to_code"},
            powl_obj=powl_model
        )
        
        if not success:
            raise Exception(f"Failed to convert POWL model to code: {str(error)}")
        
        # Define workflow for review and improvement
        workflow = [
            {
                "id": "review_event_log_model",
                "agent": "model_reviewer",
                "description": f"Review the POWL model generated from event log {event_log_path}:\n\n```python\n{powl_code}\n```\n\nProvide detailed feedback for improvement.",
                "expected_output": "Detailed review feedback",
                "context": {"powl_code": powl_code, "source": "event_log"}
            },
            {
                "id": "improve_event_log_model",
                "agent": "powl_modeler",
                "description": "Improve the POWL model based on the review feedback",
                "expected_output": "Improved POWL model code",
                "dependencies": ["review_event_log_model"],
                "context": {"powl_code": powl_code}
            }
        ]
        
        # Create and execute workflow
        self.master_agent.create_workflow(workflow)
        results = self.master_agent.execute_workflow()
        
        # Extract improved model
        improved_powl_code = None
        if "improve_event_log_model" in results and results["improve_event_log_model"]["status"] == "completed":
            improved_powl_code = results["improve_event_log_model"]["result"]
        else:
            # Use original code if improvement failed
            improved_powl_code = powl_code
        
        # Validate and fix the code if needed
        validation_result = validate_and_fix_powl_model(improved_powl_code)
        if validation_result["status"] in ["success", "warning"]:
            improved_powl_code = validation_result["fixed_code"]
        
        # Store results for reference
        self.results = {
            "event_log_path": event_log_path,
            "workflow_results": results,
            "validation_result": validation_result
        }
        
        # Create result object
        return ProcessModelResult(
            powl_code=improved_powl_code,
            api_key=self.api_key,
            ai_provider=self.ai_provider,
            model_name=self.model_name,
            original_description=f"Event log-based model from {event_log_path}",
            analysis="Discovered from event log",
            review=str(results.get("review_event_log_model", {}).get("result", ""))
        )
    
    def generate_model_from_petri_net(self, petri_net_path: str) -> ProcessModelResult:
        """
        Generate process model from Petri net with enhanced error handling
        
        Args:
            petri_net_path: Path to the Petri net file
            
        Returns:
            ProcessModelResult object
        """
        # Convert Petri net to POWL with error handling
        success, powl_model, error = safe_execute(
            convert_petri_net_to_powl,
            error_context={"operation": "convert_petri_net_to_powl"},
            petri_net_path=petri_net_path
        )
        
        if not success:
            raise Exception(f"Failed to convert Petri net to POWL: {str(error)}")
        
        # Convert POWL to code with error handling
        success, powl_code, error = safe_execute(
            powl_to_code,
            error_context={"operation": "powl_to_code"},
            powl_obj=powl_model
        )
        
        if not success:
            raise Exception(f"Failed to convert POWL model to code: {str(error)}")
        
        # Define workflow for review and improvement
        workflow = [
            {
                "id": "review_petri_net_model",
                "agent": "model_reviewer",
                "description": f"Review the POWL model generated from Petri net {petri_net_path}:\n\n```python\n{powl_code}\n```\n\nProvide detailed feedback for improvement.",
                "expected_output": "Detailed review feedback",
                "context": {"powl_code": powl_code, "source": "petri_net"}
            },
            {
                "id": "improve_petri_net_model",
                "agent": "powl_modeler",
                "description": "Improve the POWL model based on the review feedback",
                "expected_output": "Improved POWL model code",
                "dependencies": ["review_petri_net_model"],
                "context": {"powl_code": powl_code}
            }
        ]
        
        # Create and execute workflow
        self.master_agent.create_workflow(workflow)
        results = self.master_agent.execute_workflow()
        
        # Extract improved model
        improved_powl_code = None
        if "improve_petri_net_model" in results and results["improve_petri_net_model"]["status"] == "completed":
            improved_powl_code = results["improve_petri_net_model"]["result"]
        else:
            # Use original code if improvement failed
            improved_powl_code = powl_code
        
        # Validate and fix the code if needed
        validation_result = validate_and_fix_powl_model(improved_powl_code)
        if validation_result["status"] in ["success", "warning"]:
            improved_powl_code = validation_result["fixed_code"]
        
        # Store results for reference
        self.results = {
            "petri_net_path": petri_net_path,
            "workflow_results": results,
            "validation_result": validation_result
        }
        
        # Create result object
        return ProcessModelResult(
            powl_code=improved_powl_code,
            api_key=self.api_key,
            ai_provider=self.ai_provider,
            model_name=self.model_name,
            original_description=f"Petri net-based model from {petri_net_path}",
            analysis="Converted from Petri net",
            review=str(results.get("review_petri_net_model", {}).get("result", ""))
        )
    
    def generate_model_from_bpmn(self, bpmn_path: str) -> ProcessModelResult:
        """
        Generate process model from BPMN with enhanced error handling
        
        Args:
            bpmn_path: Path to the BPMN file
            
        Returns:
            ProcessModelResult object
        """
        # Convert BPMN to POWL with error handling
        success, powl_model, error = safe_execute(
            convert_bpmn_to_powl,
            error_context={"operation": "convert_bpmn_to_powl"},
            bpmn_path=bpmn_path
        )
        
        if not success:
            raise Exception(f"Failed to convert BPMN to POWL: {str(error)}")
        
        # Convert POWL to code with error handling
        success, powl_code, error = safe_execute(
            powl_to_code,
            error_context={"operation": "powl_to_code"},
            powl_obj=powl_model
        )
        
        if not success:
            raise Exception(f"Failed to convert POWL model to code: {str(error)}")
        
        # Define workflow for review and improvement
        workflow = [
            {
                "id": "review_bpmn_model",
                "agent": "model_reviewer",
                "description": f"Review the POWL model generated from BPMN {bpmn_path}:\n\n```python\n{powl_code}\n```\n\nProvide detailed feedback for improvement.",
                "expected_output": "Detailed review feedback",
                "context": {"powl_code": powl_code, "source": "bpmn"}
            },
            {
                "id": "improve_bpmn_model",
                "agent": "powl_modeler",
                "description": "Improve the POWL model based on the review feedback",
                "expected_output": "Improved POWL model code",
                "dependencies": ["review_bpmn_model"],
                "context": {"powl_code": powl_code}
            }
        ]
        
        # Create and execute workflow
        self.master_agent.create_workflow(workflow)
        results = self.master_agent.execute_workflow()
        
        # Extract improved model
        improved_powl_code = None
        if "improve_bpmn_model" in results and results["improve_bpmn_model"]["status"] == "completed":
            improved_powl_code = results["improve_bpmn_model"]["result"]
        else:
            # Use original code if improvement failed
            improved_powl_code = powl_code
        
        # Validate and fix the code if needed
        validation_result = validate_and_fix_powl_model(improved_powl_code)
        if validation_result["status"] in ["success", "warning"]:
            improved_powl_code = validation_result["fixed_code"]
        
        # Store results for reference
        self.results = {
            "bpmn_path": bpmn_path,
            "workflow_results": results,
            "validation_result": validation_result
        }
        
        # Create result object
        return ProcessModelResult(
            powl_code=improved_powl_code,
            api_key=self.api_key,
            ai_provider=self.ai_provider,
            model_name=self.model_name,
            original_description=f"BPMN-based model from {bpmn_path}",
            analysis="Converted from BPMN",
            review=str(results.get("review_bpmn_model", {}).get("result", ""))
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors and recovery attempts
        
        Returns:
            Dictionary with error statistics
        """
        return error_handler.get_error_statistics()
