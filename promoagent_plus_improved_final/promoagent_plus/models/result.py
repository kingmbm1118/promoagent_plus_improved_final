"""
Result model for ProMoAgent+ process modeling
"""

from typing import Optional, Dict, Any, Tuple
import pm4py
from pm4py.objects.powl.obj import POWL

from promoagent_plus.core.powl_utils import execute_powl_code
from promoagent_plus.agents.agent_factory import create_agents
from promoagent_plus.tasks.task_definitions import create_user_feedback_task
from crewai import Crew, Process

class ProcessModelResult:
    """Class to represent the result of a process model generation"""
    
    def __init__(
        self, 
        powl_code: str, 
        api_key: str, 
        ai_provider: str,
        model_name: str,
        original_description: str,
        analysis: str,
        review: str
    ):
        self.powl_code = powl_code
        self.api_key = api_key
        self.ai_provider = ai_provider
        self.model_name = model_name
        self.original_description = original_description
        self.analysis = analysis
        self.review = review
        
        # Parse the POWL code to get the model
        try:
            self.powl_model = execute_powl_code(powl_code)
        except Exception as e:
            print(f"Error parsing POWL code: {e}")
            self.powl_model = None
    
    def update(self, feedback: str) -> None:
        """
        Update the model based on user feedback
        
        Args:
            feedback: User feedback on the model
        """
        # Create agents
        agents = create_agents(self.api_key, self.ai_provider, self.model_name)
        
        # Create feedback task
        feedback_task = create_user_feedback_task(
            agents["powl_modeler"], 
            self.powl_code, 
            feedback, 
            self.original_description
        )
        
        # Execute task with crew
        crew = Crew(
            agents=[agents["powl_modeler"]],
            tasks=[feedback_task],
            verbose=True,
            process=Process.sequential
        )
        
        # Update model with result
        improved_powl_code = crew.kickoff()
        self.powl_code = improved_powl_code
        
        # Parse the updated POWL code
        try:
            self.powl_model = execute_powl_code(improved_powl_code)
        except Exception as e:
            print(f"Error parsing updated POWL code: {e}")
    
    def get_powl(self) -> Optional[POWL]:
        """
        Get the POWL model
        
        Returns:
            POWL model if available, None otherwise
        """
        return self.powl_model
    
    def get_petri_net(self) -> Tuple:
        """
        Get the Petri net representation of the model
        
        Returns:
            Tuple of (net, initial_marking, final_marking)
        """
        if self.powl_model is None:
            raise ValueError("No valid POWL model available")
        
        return pm4py.convert_to_petri_net(self.powl_model)
    
    def get_bpmn(self) -> Any:
        """
        Get the BPMN representation of the model
        
        Returns:
            BPMN model
        """
        if self.powl_model is None:
            raise ValueError("No valid POWL model available")
        
        net, im, fm = self.get_petri_net()
        return pm4py.convert_to_bpmn(net, im, fm)
    
    def view_powl(self, image_format: str = "svg") -> None:
        """
        Visualize the POWL model
        
        Args:
            image_format: Image format for visualization
        """
        if self.powl_model is None:
            raise ValueError("No valid POWL model available")
        
        pm4py.view_powl(self.powl_model, format=image_format)
    
    def view_petri_net(self, image_format: str = "svg") -> None:
        """
        Visualize the Petri net representation
        
        Args:
            image_format: Image format for visualization
        """
        net, im, fm = self.get_petri_net()
        pm4py.view_petri_net(net, im, fm, format=image_format)
    
    def view_bpmn(self, image_format: str = "svg") -> None:
        """
        Visualize the BPMN representation
        
        Args:
            image_format: Image format for visualization
        """
        bpmn_model = self.get_bpmn()
        pm4py.view_bpmn(bpmn_model, format=image_format)
    
    def export_powl_code(self, file_path: str) -> None:
        """
        Export the POWL code to a file
        
        Args:
            file_path: Path to save the code
        """
        with open(file_path, 'w') as f:
            f.write(self.powl_code)
    
    def export_bpmn(self, file_path: str) -> None:
        """
        Export the BPMN model to a file
        
        Args:
            file_path: Path to save the BPMN model
        """
        if not file_path.lower().endswith(".bpmn"):
            raise ValueError("The provided file path does not have the '.bpmn' extension!")
        
        bpmn_model = self.get_bpmn()
        pm4py.write_bpmn(bpmn_model, file_path)
    
    def export_petri_net(self, file_path: str) -> None:
        """
        Export the Petri net model to a file
        
        Args:
            file_path: Path to save the Petri net model
        """
        if not file_path.lower().endswith(".pnml"):
            raise ValueError("The provided file path does not have the '.pnml' extension!")
        
        net, im, fm = self.get_petri_net()
        pm4py.write_pnml(net, im, fm, file_path)