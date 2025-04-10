"""
Enhanced validation utilities for ProMoAgent+
"""

import re
import ast
from typing import Dict, List, Any, Optional, Tuple, Union
from pm4py.objects.powl.obj import POWL, StrictPartialOrder, OperatorPOWL, Operator

def fix_code_indentation(code: str) -> str:
    """
    Fix indentation issues in POWL code
    
    Args:
        code: POWL code to fix
        
    Returns:
        Fixed code with proper indentation
    """
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Remove any leading/trailing whitespace
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append("")
            continue
        
        # Determine correct indentation level based on content
        if stripped.startswith(("import ", "from ", "gen = ")):
            # Top-level statements should have no indentation
            fixed_lines.append(stripped)
        elif stripped.startswith(("def ", "class ")):
            # Function and class definitions should have no indentation
            fixed_lines.append(stripped)
        elif any(stripped.startswith(kw) for kw in ["if ", "else:", "elif ", "for ", "while ", "try:", "except:", "finally:"]):
            # Control flow statements should have appropriate indentation
            # This is simplified - would need to track current indentation level
            fixed_lines.append("    " + stripped)
        else:
            # Other statements get standard indentation
            fixed_lines.append(stripped)
    
    return '\n'.join(fixed_lines)

def check_code_completeness(code: str) -> Dict[str, Any]:
    """
    Check if POWL code is complete with required elements
    
    Args:
        code: POWL code to check
        
    Returns:
        Dictionary with completeness check results
    """
    required_elements = {
        "imports": False,
        "model_generator": False,
        "final_model": False
    }
    
    missing_elements = []
    
    # Check for imports
    if "from promoagent_plus.core.model_generator import ModelGenerator" in code:
        required_elements["imports"] = True
    else:
        missing_elements.append("Missing import: from promoagent_plus.core.model_generator import ModelGenerator")
    
    # Check for ModelGenerator initialization
    if "gen = ModelGenerator()" in code:
        required_elements["model_generator"] = True
    else:
        missing_elements.append("Missing ModelGenerator initialization: gen = ModelGenerator()")
    
    # Check for final_model assignment
    if "final_model =" in code:
        required_elements["final_model"] = True
    else:
        missing_elements.append("Missing final model assignment: final_model = ...")
    
    return {
        "is_complete": all(required_elements.values()),
        "missing_elements": missing_elements
    }

def check_syntax_issues(code: str) -> List[str]:
    """
    Check for common syntax issues in POWL code
    
    Args:
        code: POWL code to check
        
    Returns:
        List of identified syntax issues
    """
    issues = []
    
    try:
        # Try to parse the code with ast
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return issues
    
    # Check for common issues
    lines = code.split('\n')
    
    # Check for missing parentheses in function calls
    for i, line in enumerate(lines):
        if "gen." in line and "=" in line and "(" not in line:
            issues.append(f"Missing parentheses in function call at line {i+1}: {line.strip()}")
    
    # Check for missing commas in function arguments
    for i, line in enumerate(lines):
        if "gen." in line and "(" in line and ")" in line:
            args_part = line.split("(", 1)[1].split(")", 1)[0]
            if args_part.strip() and "," not in args_part and " " in args_part.strip():
                issues.append(f"Possible missing comma in function arguments at line {i+1}: {line.strip()}")
    
    return issues

def get_syntax_suggestions(issues: List[str]) -> List[str]:
    """
    Get suggestions for fixing syntax issues
    
    Args:
        issues: List of syntax issues
        
    Returns:
        List of suggestions
    """
    suggestions = []
    
    for issue in issues:
        if "missing parentheses" in issue.lower():
            suggestions.append("Add parentheses to function calls: gen.method()")
        elif "missing comma" in issue.lower():
            suggestions.append("Separate function arguments with commas: func(arg1, arg2)")
        elif "syntax error" in issue.lower():
            suggestions.append("Check for proper syntax, including brackets, quotes, and indentation")
        else:
            suggestions.append(f"Fix the issue: {issue}")
    
    # Add general suggestions
    if suggestions:
        suggestions.append("Ensure proper indentation throughout the code")
        suggestions.append("Check for balanced parentheses and brackets")
    
    return suggestions

def get_code_fragment(code: str, line_no: Union[int, str]) -> str:
    """
    Get a code fragment around the specified line number
    
    Args:
        code: The code to extract from
        line_no: Line number (or string representation)
        
    Returns:
        Code fragment with context
    """
    try:
        if isinstance(line_no, str):
            line_no = int(line_no)
    except ValueError:
        line_no = 1
    
    lines = code.split('\n')
    start_line = max(0, line_no - 2)
    end_line = min(len(lines), line_no + 2)
    
    fragment_lines = []
    for i in range(start_line, end_line):
        prefix = ">>> " if i == line_no - 1 else "    "
        if i < len(lines):
            fragment_lines.append(f"{prefix}{lines[i]}")
    
    return '\n'.join(fragment_lines)

def validate_powl_structure(model: POWL) -> List[str]:
    """
    Validate POWL model structure
    
    Args:
        model: POWL model to validate
        
    Returns:
        List of structure issues
    """
    issues = []
    
    # Check for empty model
    if model is None:
        issues.append("Model is empty")
        return issues
    
    # Validate structure based on model type
    if isinstance(model, OperatorPOWL):
        # Check XOR operator
        if model.operator == Operator.XOR:
            if len(model.children) < 2:
                issues.append(f"XOR operator has less than 2 children ({len(model.children)})")
        
        # Check LOOP operator
        elif model.operator == Operator.LOOP:
            if len(model.children) != 2:
                issues.append(f"LOOP operator should have exactly 2 children (do and redo), but has {len(model.children)}")
    
    # Check partial order
    elif isinstance(model, StrictPartialOrder):
        # Check for empty partial order
        if len(model.get_children()) == 0:
            issues.append("Partial order has no children")
        
        # Check for irreflexivity
        if not model.order.is_irreflexive():
            issues.append("Partial order relation is not irreflexive")
    
    # Recursively validate children
    if hasattr(model, 'children'):
        for i, child in enumerate(model.children):
            child_issues = validate_powl_structure(child)
            for issue in child_issues:
                issues.append(f"Child {i} issue: {issue}")
    
    return issues

def get_structure_suggestions(issues: List[str]) -> List[str]:
    """
    Get suggestions for fixing structure issues
    
    Args:
        issues: List of structure issues
        
    Returns:
        List of suggestions
    """
    suggestions = []
    
    for issue in issues:
        if "XOR operator has less than 2 children" in issue:
            suggestions.append("Ensure XOR operators have at least 2 branches")
            suggestions.append("Add a default branch to XOR if needed")
        elif "LOOP operator should have exactly 2 children" in issue:
            suggestions.append("Ensure LOOP operators have exactly 2 children (do and redo parts)")
        elif "Partial order has no children" in issue:
            suggestions.append("Add activities to the partial order")
        elif "not irreflexive" in issue:
            suggestions.append("Remove self-loops in partial order relations")
        else:
            suggestions.append(f"Address the structure issue: {issue}")
    
    return suggestions

def get_type_error_suggestions(error_msg: str) -> List[str]:
    """
    Get suggestions for fixing type errors
    
    Args:
        error_msg: Type error message
        
    Returns:
        List of suggestions
    """
    suggestions = []
    
    if "has no attribute" in error_msg:
        attr = error_msg.split("has no attribute")[1].strip().strip("'").strip('"')
        suggestions.append(f"Ensure the object has the '{attr}' attribute before accessing it")
        suggestions.append("Check for typos in attribute names")
        suggestions.append("Convert the object to the appropriate type if needed")
    
    elif "object is not callable" in error_msg:
        suggestions.append("Ensure you're calling a function or method, not accessing a property")
        suggestions.append("Check for parentheses usage in function calls")
    
    elif "takes" in error_msg and "positional argument" in error_msg:
        suggestions.append("Check the number of arguments passed to functions")
        suggestions.append("Ensure required arguments are provided")
        suggestions.append("Use keyword arguments for clarity")
    
    else:
        suggestions.append("Check variable types and ensure they match expected types")
        suggestions.append("Use explicit type conversions when needed")
        suggestions.append("Verify object attributes and methods before use")
    
    return suggestions

def get_model_summary(model: POWL) -> Dict[str, Any]:
    """
    Get a summary of a POWL model
    
    Args:
        model: POWL model to summarize
        
    Returns:
        Dictionary with model summary
    """
    summary = {
        "type": type(model).__name__,
        "activities": 0,
        "silent_transitions": 0,
        "xor_operators": 0,
        "loop_operators": 0,
        "partial_orders": 0
    }
    
    def count_elements(node):
        if node is None:
            return
        
        if isinstance(node, OperatorPOWL):
            if node.operator == Operator.XOR:
                summary["xor_operators"] += 1
            elif node.operator == Operator.LOOP:
                summary["loop_operators"] += 1
            
            for child in node.children:
                count_elements(child)
        
        elif isinstance(node, StrictPartialOrder):
            summary["partial_orders"] += 1
            for child in node.get_children():
                count_elements(child)
        
        else:  # Transition or SilentTransition
            if hasattr(node, 'label') and node.label:
                summary["activities"] += 1
            else:
                summary["silent_transitions"] += 1
    
    count_elements(model)
    
    return summary

def auto_fix_powl_code(code: str, error_msg: str = None) -> Tuple[str, List[str]]:
    """
    Automatically fix common issues in POWL code
    
    Args:
        code: POWL code to fix
        error_msg: Optional error message to guide fixes
        
    Returns:
        Tuple of (fixed_code, applied_fixes)
    """
    applied_fixes = []
    fixed_code = code
    
    # Fix 1: Fix indentation
    fixed_code = fix_code_indentation(fixed_code)
    applied_fixes.append("Fixed code indentation")
    
    # Fix 2: Add missing imports
    if "from promoagent_plus.core.model_generator import ModelGenerator" not in fixed_code:
        fixed_code = "from promoagent_plus.core.model_generator import ModelGenerator\n" + fixed_code
        applied_fixes.append("Added missing import for ModelGenerator")
    
    # Fix 3: Add ModelGenerator initialization
    if "gen = ModelGenerator()" not in fixed_code:
        # Find the right place to add it (after imports)
        lines = fixed_code.split("\n")
        import_end_idx = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")) or not line.strip():
                import_end_idx = i
            else:
                break
        
        # Insert after imports
        lines.insert(import_end_idx + 1, "gen = ModelGenerator()")
        fixed_code = "\n".join(lines)
        applied_fixes.append("Added ModelGenerator initialization")
    
    # Fix 4: Fix XOR with less than 2 submodels
    if error_msg and "Cannot create an xor of less than 2 submodels" in error_msg:
        lines = fixed_code.split("\n")
        
        for i, line in enumerate(lines):
            if "gen.xor(" in line:
                # Count the number of arguments
                args_start = line.find("(") + 1
                args_end = line.rfind(")")
                if args_start < args_end:
                    args = line[args_start:args_end].split(",")
                    
                    # If there's only one argument, add a silent transition
                    if len(args) < 2:
                        new_line = line[:args_end] + ", gen.silent_transition()" + line[args_end:]
                        lines[i] = new_line
                        applied_fixes.append("Added silent transition to XOR with less than 2 submodels")
        
        fixed_code = "\n".join(lines)
    
    # Fix 5: Ensure final_model is assigned
    if "final_model =" not in fixed_code:
        # Find the last variable assignment
        last_var = None
        for line in fixed_code.split("\n"):
            if "=" in line and "var_" in line.split("=")[0]:
                last_var = line.split("=")[0].strip()
        
        if last_var:
            fixed_code += f"\nfinal_model = {last_var}"
            applied_fixes.append("Added missing final_model assignment")
    
    # Fix 6: Fix 'CrewOutput' object has no attribute 'split' error
    if error_msg and "'CrewOutput' object has no attribute 'split'" in error_msg:
        # Find the line that's trying to split a CrewOutput
        lines = fixed_code.split("\n")
        for i, line in enumerate(lines):
            if ".split" in line:
                # Replace with str() conversion
                lines[i] = lines[i].replace(".split", ".__str__().split")
                applied_fixes.append("Added string conversion for CrewOutput object")
        
        fixed_code = "\n".join(lines)
    
    return fixed_code, applied_fixes

def validate_and_fix_powl_model(powl_code: str) -> Dict[str, Any]:
    """
    Validate and fix POWL model code
    
    Args:
        powl_code: POWL code to validate and fix
        
    Returns:
        Dictionary with validation results and fixed code
    """
    from promoagent_plus.core.powl_utils import execute_powl_code
    
    # Initial validation
    try:
        # Fix indentation issues
        fixed_code = fix_code_indentation(powl_code)
        
        # Check for completeness
        completeness_check = check_code_completeness(fixed_code)
        if not completeness_check["is_complete"]:
            # Try to auto-fix completeness issues
            fixed_code, applied_fixes = auto_fix_powl_code(fixed_code)
            
            # Check again after fixes
            completeness_check = check_code_completeness(fixed_code)
            if not completeness_check["is_complete"]:
                return {
                    "status": "error",
                    "error_type": "incomplete_code",
                    "message": "The code is incomplete",
                    "details": completeness_check["missing_elements"],
                    "fixed_code": fixed_code,
                    "applied_fixes": applied_fixes
                }
        
        # Check for common syntax issues
        syntax_issues = check_syntax_issues(fixed_code)
        if syntax_issues:
            return {
                "status": "error",
                "error_type": "syntax",
                "message": f"Syntax issues detected: {syntax_issues}",
                "suggestions": get_syntax_suggestions(syntax_issues),
                "fixed_code": fixed_code
            }
        
        # Try executing the code
        model = execute_powl_code(fixed_code)
        
        # Validate the model structure
        validation_issues = validate_powl_structure(model)
        if validation_issues:
            return {
                "status": "warning",
                "warning_type": "structure",
                "message": f"POWL model structure issues: {validation_issues}",
                "suggestions": get_structure_suggestions(validation_issues),
                "fixed_code": fixed_code,
                "model_info": get_model_summary(model)
            }
        
        # Model is valid
        return {
            "status": "success",
            "message": "The POWL model code is valid and creates a proper POWL model.",
            "model_info": get_model_summary(model),
            "fixed_code": fixed_code
        }
    
    except SyntaxError as e:
        line_no = getattr(e, 'lineno', '?')
        # Try to auto-fix the code
        fixed_code, applied_fixes = auto_fix_powl_code(powl_code, str(e))
        
        return {
            "status": "error",
            "error_type": "syntax",
            "message": f"Syntax error at line {line_no}: {str(e)}",
            "code_fragment": get_code_fragment(powl_code, line_no),
            "suggestions": [
                "Check for missing parentheses or brackets",
                "Ensure proper indentation",
                "Verify correct parameter names"
            ],
            "fixed_code": fixed_code,
            "applied_fixes": applied_fixes
        }
    
    except TypeError as e:
        # Try to auto-fix the code
        fixed_code, applied_fixes = auto_fix_powl_code(powl_code, str(e))
        
        return {
            "status": "error",
            "error_type": "type_error",
            "message": f"Type error: {str(e)}",
            "suggestions": get_type_error_suggestions(str(e)),
            "fixed_code": fixed_code,
            "applied_fixes": applied_fixes
        }
    
    except Exception as e:
        # Try to auto-fix the code
        fixed_code, applied_fixes = auto_fix_powl_code(powl_code, str(e))
        
        return {
            "status": "error",
            "error_type": "general",
            "message": f"Error validating POWL model: {str(e)}",
            "suggestions": [
                "Try simplifying the model first",
                "Ensure each submodel is used uniquely",
                "Check for proper method arguments"
            ],
            "fixed_code": fixed_code,
            "applied_fixes": applied_fixes
        }
