"""
Tools for the CrewAI agents in ProMoAgent+
"""

import json
from langchain.tools import tool
from pm4py.objects.powl.obj import POWL
import pm4py

from promoagent_plus.core.powl_utils import extract_powl_code, execute_powl_code, validate_powl_model
from promoagent_plus.utils.validation_utils import (
    check_syntax_issues, 
    get_syntax_suggestions,
    get_code_fragment,
    validate_powl_structure,
    get_structure_suggestions,
    get_type_error_suggestions,
    get_model_summary,
    fix_code_indentation,
    check_code_completeness
)


@tool
def analyze_text_complexity(text: str) -> str:
    """
    Analyzes the complexity of a process description text
    
    Args:
        text: The process description text
        
    Returns:
        JSON string with complexity metrics
    """
    word_count = len(text.split())
    sentence_count = len(text.split('.'))
    
    complexity = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": word_count / max(1, sentence_count)
    }
    
    complexity_level = "Low"
    if complexity["word_count"] > 200:
        complexity_level = "High"
    elif complexity["word_count"] > 100:
        complexity_level = "Medium"
    
    return json.dumps({
        "metrics": complexity,
        "complexity_level": complexity_level
    })

@tool
def extract_activities_from_text(text: str) -> str:
    """
    Extracts potential activities from a process description.
    This is a placeholder - in practice this would be performed by the LLM agent.
    
    Args:
        text: The process description text
        
    Returns:
        JSON string with extracted activities
    """
    return json.dumps({
        "message": "Activity extraction should be performed by the LLM agent",
        "example_format": [
            {"name": "activity_name", "description": "what the activity does"}
        ]
    })

@tool
def validate_powl_code(powl_code: str) -> str:
    """
    Validates a POWL model code by trying to execute it and checking for errors
    
    Args:
        powl_code: Python code for creating a POWL model
        
    Returns:
        Validation results with detailed feedback
    """
    try:
        # Fix indentation issues
        fixed_code = fix_code_indentation(powl_code)
        
        # Check for completeness
        completeness_check = check_code_completeness(fixed_code)
        if not completeness_check["is_complete"]:
            return json.dumps({
                "status": "error",
                "error_type": "incomplete_code",
                "message": "The code is incomplete",
                "details": completeness_check["missing_elements"],
                "suggestions": [
                    "Make sure to include proper imports",
                    "Initialize ModelGenerator with 'gen = ModelGenerator()'",
                    "Assign the final model to 'final_model'"
                ]
            })
        
        # Check for common syntax issues
        syntax_issues = check_syntax_issues(fixed_code)
        if syntax_issues:
            return json.dumps({
                "status": "error",
                "error_type": "syntax",
                "message": f"Syntax issues detected: {syntax_issues}",
                "suggestions": get_syntax_suggestions(syntax_issues)
            })
        
        # Try executing the code
        model = execute_powl_code(fixed_code)
        
        # Validate the model structure
        validation_issues = validate_powl_structure(model)
        if validation_issues:
            return json.dumps({
                "status": "warning",
                "warning_type": "structure",
                "message": f"POWL model structure issues: {validation_issues}",
                "suggestions": get_structure_suggestions(validation_issues)
            })
        
        # Model is valid
        return json.dumps({
            "status": "success",
            "message": "The POWL model code is valid and creates a proper POWL model.",
            "model_info": get_model_summary(model)
        })
    
    except SyntaxError as e:
        line_no = getattr(e, 'lineno', '?')
        return json.dumps({
            "status": "error",
            "error_type": "syntax",
            "message": f"Syntax error at line {line_no}: {str(e)}",
            "code_fragment": get_code_fragment(fixed_code, line_no),
            "suggestions": [
                "Check for missing parentheses or brackets",
                "Ensure proper indentation",
                "Verify correct parameter names"
            ]
        })
    except TypeError as e:
        return json.dumps({
            "status": "error",
            "error_type": "type_error",
            "message": f"Type error: {str(e)}",
            "suggestions": get_type_error_suggestions(str(e))
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error_type": "general",
            "message": f"Error validating POWL model: {str(e)}",
            "suggestions": [
                "Try simplifying the model first",
                "Ensure each submodel is used uniquely",
                "Check for proper method arguments"
            ]
        })

@tool
def convert_powl_to_bpmn(powl_code: str) -> str:
    """
    Converts a POWL model to BPMN representation
    
    Args:
        powl_code: Python code for creating a POWL model
        
    Returns:
        Conversion results
    """
    try:
        model = execute_powl_code(powl_code)
        
        # Convert to Petri net
        net, im, fm = pm4py.convert_to_petri_net(model)
        
        # Convert to BPMN
        bpmn_model = pm4py.convert_to_bpmn(net, im, fm)
        
        return "Successfully converted POWL to BPMN model."
    except Exception as e:
        return f"Error converting to BPMN: {str(e)}"

@tool
def convert_powl_to_petri_net(powl_code: str) -> str:
    """
    Converts a POWL model to Petri net representation
    
    Args:
        powl_code: Python code for creating a POWL model
        
    Returns:
        Conversion results
    """
    try:
        model = execute_powl_code(powl_code)
        
        # Convert to Petri net
        net, im, fm = pm4py.convert_to_petri_net(model)
        
        return "Successfully converted POWL to Petri net model."
    except Exception as e:
        return f"Error converting to Petri net: {str(e)}"

@tool
def evaluate_model_quality(powl_code: str, description: str) -> str:
    """
    Evaluates the quality of a POWL model against a process description
    This is mainly a placeholder as the actual evaluation would be performed by the LLM
    
    Args:
        powl_code: Python code for creating a POWL model
        description: Original process description
        
    Returns:
        JSON string with quality metrics
    """
    try:
        # Just check if the model is valid
        model = execute_powl_code(powl_code)
        validate_powl_model(model)
        
        # In a real implementation, we would compute actual metrics
        return json.dumps({
            "message": "Model quality evaluation should be performed by the LLM agent",
            "example_metrics": {
                "completeness": 0.85,
                "correctness": 0.90,
                "simplicity": 0.80
            }
        })
    except Exception as e:
        return json.dumps({
            "error": f"Error evaluating model quality: {str(e)}"
        })