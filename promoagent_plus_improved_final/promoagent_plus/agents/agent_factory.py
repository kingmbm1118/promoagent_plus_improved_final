"""
Modified agent_factory.py to use OpenAI models and fix tool format compatibility
"""

from typing import Dict, List, Any
from crewai import Agent
from langchain.tools import BaseTool
from langchain.tools.base import StructuredTool

from promoagent_plus.utils.constants import AIProviders, AI_MODEL_DEFAULTS
from promoagent_plus.agents.tools import (
    analyze_text_complexity,
    extract_activities_from_text,
    validate_powl_code,
    convert_powl_to_bpmn,
    convert_powl_to_petri_net,
    evaluate_model_quality
)

def get_llm_for_agent(api_key: str, ai_provider: str, model_name: str = None):
    """
    Returns appropriate LLM instance based on provider
    
    Args:
        api_key: API key for the LLM provider
        ai_provider: The AI provider to use
        model_name: Optional specific model name to use
        
    Returns:
        LLM instance
    """
    if model_name is None:
        model_name = AI_MODEL_DEFAULTS[ai_provider]
    
    # Use OpenAI for all providers for compatibility
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4", openai_api_key=api_key)

def convert_to_base_tool(tool):
    """
    Convert a StructuredTool to a BaseTool for compatibility with crewai
    
    Args:
        tool: The tool to convert
        
    Returns:
        BaseTool instance
    """
    if isinstance(tool, StructuredTool):
        # Create a BaseTool with the same properties
        base_tool = BaseTool(
            name=tool.name,
            description=tool.description,
            func=tool._run,
            coroutine=tool._acall if hasattr(tool, '_acall') else None,
            return_direct=tool.return_direct
        )
        return base_tool
    return tool

def create_agents(api_key: str, ai_provider: str, model_name: str = None) -> Dict[str, Agent]:
    """
    Create a set of specialized agents for ProMoAgent+
    
    Args:
        api_key: API key for the LLM provider
        ai_provider: The AI provider to use
        model_name: Optional specific model name to use
        
    Returns:
        Dictionary of agent instances
    """
    # Convert tools to BaseTool format for compatibility
    base_analyze_tool = convert_to_base_tool(analyze_text_complexity)
    base_extract_tool = convert_to_base_tool(extract_activities_from_text)
    base_validate_tool = convert_to_base_tool(validate_powl_code)
    base_convert_bpmn_tool = convert_to_base_tool(convert_powl_to_bpmn)
    base_convert_petri_tool = convert_to_base_tool(convert_powl_to_petri_net)
    base_evaluate_tool = convert_to_base_tool(evaluate_model_quality)
    
    # Process Analyzer Agent
    process_analyzer = Agent(
        role="Process Analyzer",
        goal="Analyze process descriptions and identify activities, actors, and control flow",
        backstory="As a process analysis expert, I can identify the key components of a business process from textual descriptions. I understand industry standard terminology and can extract structured information from natural language.",
        verbose=True,
        allow_delegation=True,
        tools=[base_analyze_tool, base_extract_tool],
        llm=get_llm_for_agent(api_key, ai_provider, model_name)
    )
    
    # POWL Modeler Agent
    powl_modeler = Agent(
        role="POWL Modeler",
        goal="Create precise POWL models based on process components and structure",
        backstory="I am an expert in process modeling with deep knowledge of POWL language. I understand how to properly structure processes with XOR choices, loops, and partial orders to accurately represent business logic.",
        verbose=True,
        allow_delegation=True,
        tools=[base_validate_tool],
        llm=get_llm_for_agent(api_key, ai_provider, model_name)
    )
    
    # Model Reviewer Agent
    model_reviewer = Agent(
        role="Model Reviewer",
        goal="Verify model correctness and suggest improvements",
        backstory="I review process models to ensure they correctly represent the described process. I look for logical errors, missing elements, and opportunities for optimization while ensuring the model remains faithful to the requirements.",
        verbose=True,
        allow_delegation=True,
        tools=[base_validate_tool, base_evaluate_tool],
        llm=get_llm_for_agent(api_key, ai_provider, model_name)
    )
    
    # Model Translator Agent
    model_translator = Agent(
        role="Model Translator",
        goal="Convert POWL models to target formats like BPMN and Petri nets",
        backstory="I specialize in translating between different process modeling notations while preserving semantic equivalence. I understand the strengths and limitations of each notation and ensure the translation maintains the original process logic.",
        verbose=True,
        allow_delegation=True,
        tools=[base_convert_bpmn_tool, base_convert_petri_tool],
        llm=get_llm_for_agent(api_key, ai_provider, model_name)
    )
    
    return {
        "process_analyzer": process_analyzer,
        "powl_modeler": powl_modeler,
        "model_reviewer": model_reviewer,
        "model_translator": model_translator
    }
