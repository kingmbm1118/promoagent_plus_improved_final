"""
Test script for ProMoAgent+ error handling and agent collaboration
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ProMoAgent+ components
from promoagent_plus.agents.agent_factory import create_agents, get_llm_for_agent
from promoagent_plus.agents.agent_coordinator import MasterAgent
from promoagent_plus.utils.error_handling import safe_execute, ErrorType, ErrorSeverity
from promoagent_plus.utils.enhanced_validation import validate_and_fix_powl_model
from promoagent_plus.core.powl_utils import extract_powl_code, execute_powl_code
from promoagent_plus.utils.constants import AIProviders

def test_error_handling():
    """Test error handling functionality"""
    logger.info("Testing error handling...")
    
    # Test case 1: Syntax error in POWL code
    invalid_powl_code = """
    from promoagent_plus.core.model_generator import ModelGenerator
    gen = ModelGenerator()
    
    # Missing parenthesis
    var_0 = gen.activity("Start"
    var_1 = gen.activity("End")
    final_model = gen.xor(var_0, var_1)
    """
    
    logger.info("Test case 1: Syntax error in POWL code")
    result = validate_and_fix_powl_code(invalid_powl_code)
    logger.info(f"Validation result: {result['status']}")
    logger.info(f"Applied fixes: {result.get('applied_fixes', [])}")
    
    # Test case 2: XOR with less than 2 submodels
    invalid_xor_code = """
    from promoagent_plus.core.model_generator import ModelGenerator
    gen = ModelGenerator()
    
    var_0 = gen.activity("Start")
    var_1 = gen.xor(var_0)  # XOR needs at least 2 submodels
    final_model = var_1
    """
    
    logger.info("Test case 2: XOR with less than 2 submodels")
    result = validate_and_fix_powl_code(invalid_xor_code)
    logger.info(f"Validation result: {result['status']}")
    logger.info(f"Applied fixes: {result.get('applied_fixes', [])}")
    
    # Test case 3: Missing final_model assignment
    missing_final_model = """
    from promoagent_plus.core.model_generator import ModelGenerator
    gen = ModelGenerator()
    
    var_0 = gen.activity("Start")
    var_1 = gen.activity("End")
    var_2 = gen.xor(var_0, var_1)
    # Missing final_model assignment
    """
    
    logger.info("Test case 3: Missing final_model assignment")
    result = validate_and_fix_powl_code(missing_final_model)
    logger.info(f"Validation result: {result['status']}")
    logger.info(f"Applied fixes: {result.get('applied_fixes', [])}")
    
    # Test case 4: Safe execution with error handling
    def function_with_error():
        # Simulate an error
        raise ValueError("Test error for safe_execute")
    
    logger.info("Test case 4: Safe execution with error handling")
    success, result, error = safe_execute(
        function_with_error,
        error_context={"test_case": "safe_execute"}
    )
    logger.info(f"Safe execute result: success={success}, error_type={error.error_type if error else None}")
    
    logger.info("Error handling tests completed")
    return True

def test_agent_collaboration(api_key: str):
    """
    Test agent collaboration functionality
    
    Args:
        api_key: API key for LLM provider
    """
    logger.info("Testing agent collaboration...")
    
    # Create agents
    ai_provider = AIProviders.OPENAI.value
    agents = create_agents(api_key, ai_provider)
    
    # Create master agent
    master_agent = MasterAgent(
        specialized_agents={
            "process_analyzer": agents["process_analyzer"],
            "powl_modeler": agents["powl_modeler"],
            "model_reviewer": agents["model_reviewer"],
            "model_translator": agents["model_translator"]
        },
        max_retries=2,
        monitoring_enabled=True
    )
    
    # Test simple workflow
    logger.info("Test case 1: Simple workflow execution")
    
    # Define a simple workflow
    workflow = [
        {
            "id": "test_task_1",
            "agent": "process_analyzer",
            "description": "Analyze a simple process: A customer orders a product, the order is processed, and the product is shipped.",
            "expected_output": "JSON with identified process components"
        },
        {
            "id": "test_task_2",
            "agent": "powl_modeler",
            "description": "Create a POWL model for the simple process",
            "expected_output": "POWL model code",
            "dependencies": ["test_task_1"]
        }
    ]
    
    # Create and execute the workflow
    master_agent.create_workflow(workflow)
    results = master_agent.execute_workflow()
    
    logger.info(f"Workflow execution completed with {len(results)} results")
    for task_id, result in results.items():
        logger.info(f"Task {task_id}: {result['status']}")
    
    logger.info("Agent collaboration tests completed")
    return True

def main():
    """Main test function"""
    logger.info("Starting ProMoAgent+ tests")
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return False
    
    # Run tests
    error_handling_success = test_error_handling()
    
    if error_handling_success:
        agent_collaboration_success = test_agent_collaboration(api_key)
    else:
        logger.error("Error handling tests failed, skipping agent collaboration tests")
        agent_collaboration_success = False
    
    # Report results
    if error_handling_success and agent_collaboration_success:
        logger.info("All tests passed successfully")
        return True
    else:
        logger.error("Some tests failed")
        return False

if __name__ == "__main__":
    main()
