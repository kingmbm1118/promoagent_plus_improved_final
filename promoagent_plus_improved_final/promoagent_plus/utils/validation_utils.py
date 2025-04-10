"""
Utility functions for POWL code validation and feedback
"""

import json
import re
from typing import List, Dict, Any, Optional

def check_syntax_issues(code: str) -> list:
    """Check for common syntax issues in the code."""
    issues = []
    
    # Check for mismatched parentheses/brackets
    if code.count('(') != code.count(')'):
        issues.append("Mismatched parentheses")
    if code.count('[') != code.count(']'):
        issues.append("Mismatched square brackets")
    
    # Check for common POWL errors
    if "gen.xor(choices=" in code:
        issues.append("Incorrect parameter 'choices' in xor() function")
    
    # Check for missing final_model
    if "final_model" not in code:
        issues.append("Missing 'final_model' assignment")
    
    return issues

def get_syntax_suggestions(issues: list) -> list:
    """Get suggestions to fix syntax issues."""
    suggestions = []
    
    mapping = {
        "Mismatched parentheses": "Check that all opening '(' have matching closing ')'",
        "Mismatched square brackets": "Check that all opening '[' have matching closing ']'",
        "Incorrect parameter 'choices' in xor() function": "The xor() function doesn't use 'choices=' parameter. Use gen.xor(confirm_payment, payment_issue) instead.",
        "Missing 'final_model' assignment": "Ensure your code assigns the final model to variable 'final_model'"
    }
    
    for issue in issues:
        if issue in mapping:
            suggestions.append(mapping[issue])
    
    return suggestions

def get_code_fragment(code: str, line_no: str) -> str:
    """Extract the relevant code fragment around the error."""
    if line_no == '?':
        return "Could not determine error location"
    
    try:
        line_no = int(line_no)
        lines = code.split("\n")
        start = max(0, line_no - 3)
        end = min(len(lines), line_no + 2)
        
        result = []
        for i in range(start, end):
            prefix = ">>> " if i == line_no - 1 else "    "
            result.append(f"{prefix}{lines[i]}")
        
        return "\n".join(result)
    except:
        return "Could not extract code fragment"

def validate_powl_structure(model) -> list:
    """Validate the structure of a POWL model for common modeling mistakes."""
    issues = []
    
    # Check model structure here - simplified example
    try:
        if hasattr(model, 'children') and len(model.children) == 0:
            issues.append("Empty model with no children")
            
        # Check for proper nesting
        check_powl_nesting(model, issues)
    except Exception as e:
        issues.append(f"Error checking model structure: {str(e)}")
    
    return issues

def check_powl_nesting(node, issues, path=None):
    """Recursively check for proper POWL model nesting."""
    if path is None:
        path = []
    
    from pm4py.objects.powl.obj import StrictPartialOrder, OperatorPOWL
    
    # Check for partial orders as children of other partial orders
    if isinstance(node, StrictPartialOrder):
        if any(isinstance(p, StrictPartialOrder) for p in path):
            issues.append("Partial order detected as child of another partial order")
    
    # Add current node to path for children
    path.append(node)
    
    # Recursively check children
    if hasattr(node, 'children'):
        for child in node.children:
            check_powl_nesting(child, issues, path.copy())

def get_structure_suggestions(issues: list) -> list:
    """Get suggestions to fix model structure issues."""
    suggestions = []
    
    mapping = {
        "Empty model with no children": "Ensure your model contains actual activities",
        "Partial order detected as child of another partial order": "Avoid nesting partial orders - flatten your model structure"
    }
    
    for issue in issues:
        for key in mapping:
            if key in issue:
                suggestions.append(mapping[key])
    
    if not suggestions:
        suggestions = [
            "Review the model structure for logical flow",
            "Ensure proper modeling of XOR choices and dependencies"
        ]
    
    return suggestions

def get_type_error_suggestions(error_msg: str) -> list:
    """Get suggestions based on TypeError message."""
    suggestions = []
    
    if "argument" in error_msg and "required" in error_msg:
        suggestions.append("Check function arguments and make sure required parameters are provided")
    
    if "xor" in error_msg:
        suggestions.append("For xor(), use: gen.xor(option1, option2, ...)")
    
    if "partial_order" in error_msg:
        suggestions.append("For partial_order(), use: gen.partial_order(dependencies=[(a, b), (b, c), ...])")
    
    if "loop" in error_msg:
        suggestions.append("For loop(), use: gen.loop(do=activity_a, redo=activity_b)")
    
    if not suggestions:
        suggestions = ["Check parameter types and function signatures"]
    
    return suggestions

def get_model_summary(model) -> dict:
    """Get a summary of the POWL model for feedback."""
    summary = {
        "type": model.__class__.__name__,
        "activities": []
    }
    
    # Extract activity information
    try:
        from pm4py.objects.powl.obj import Transition, StrictPartialOrder, OperatorPOWL
        
        def extract_activities(node):
            activities = []
            if isinstance(node, Transition) and hasattr(node, 'label'):
                activities.append(node.label)
            elif hasattr(node, 'children'):
                for child in node.children:
                    activities.extend(extract_activities(child))
            return activities
        
        summary["activities"] = extract_activities(model)
        summary["activity_count"] = len(summary["activities"])
    except Exception as e:
        summary["extraction_error"] = str(e)
    
    return summary

def get_error_specific_suggestions(error_message: str) -> str:
    """Generate specific suggestions based on error message."""
    suggestions = []
    
    # Check for common errors
    if "unexpected indent" in error_message:
        suggestions.append("- Remove any leading spaces/indentation from your code lines")
        suggestions.append("- Ensure consistent indentation throughout the code")
    
    if "choices=" in error_message:
        suggestions.append("- The xor() function doesn't accept a 'choices' parameter")
        suggestions.append("- Use gen.xor(option1, option2, ...) instead of gen.xor(choices=[...])")
    
    if "not a POWL model" in error_message:
        suggestions.append("- Ensure final_model is a valid POWL object created with ModelGenerator")
        suggestions.append("- Check that you're not assigning a string or other non-POWL type")
    
    if "Cannot create an xor of less than 2 submodels" in error_message:
        suggestions.append("- Make sure your xor() function has at least 2 arguments")
        suggestions.append("- Example: gen.xor(activity_a, activity_b)")
    
    if "unique" in error_message and "submodel" in error_message:
        suggestions.append("- Each POWL component can only be used once in the model")
        suggestions.append("- Create separate activity variables for each activity, even if they have the same label")
    
    # If no specific suggestions, provide general ones
    if not suggestions:
        suggestions = [
            "- Carefully check function parameters",
            "- Ensure proper variable definitions",
            "- Start with a simpler model and gradually add complexity",
            "- Try a completely different approach if stuck with the same error"
        ]
    
    return "\n".join(suggestions)

def fix_code_indentation(code: str) -> str:
    """Fix common indentation issues in POWL code."""
    # Remove leading spaces from each line
    lines = code.split("\n")
    cleaned_lines = [line.lstrip() for line in lines]
    
    # Fix trailing spaces
    cleaned_lines = [line.rstrip() for line in cleaned_lines]
    
    return "\n".join(cleaned_lines)

def check_code_completeness(code: str) -> Dict[str, Any]:
    """Check if the code has all necessary components."""
    result = {
        "is_complete": True,
        "missing_elements": []
    }
    
    # Check for required imports
    if "from promoagent_plus.core.model_generator import ModelGenerator" not in code:
        result["is_complete"] = False
        result["missing_elements"].append("Missing import for ModelGenerator")
    
    # Check for ModelGenerator initialization
    if not re.search(r'gen\s*=\s*ModelGenerator\(\)', code):
        result["is_complete"] = False
        result["missing_elements"].append("Missing ModelGenerator initialization")
    
    # Check for final model assignment
    if not re.search(r'final_model\s*=', code):
        result["is_complete"] = False
        result["missing_elements"].append("Missing final_model assignment")
    
    return result