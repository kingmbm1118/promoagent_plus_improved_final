"""
Error handling utilities for ProMoAgent+
"""

import re
import json
import logging
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Enum for error severity levels"""
    CRITICAL = "critical"  # Prevents model generation completely
    MAJOR = "major"        # Significantly impacts model quality
    MINOR = "minor"        # Small issues that can be automatically corrected
    WARNING = "warning"    # Non-blocking issues that should be reported


class ErrorType(Enum):
    """Enum for error types"""
    PARSING = "parsing"                # Issues with parsing LLM output format
    EXECUTION = "execution"            # Errors when executing POWL code
    VALIDATION = "validation"          # Issues with POWL model structure and validation
    AGENT_COMMUNICATION = "agent_comm" # Problems in agent interaction
    RESOURCE = "resource"              # Issues with external resources or limitations
    UNKNOWN = "unknown"                # Unclassified errors


class ModelError(Exception):
    """
    Custom exception class for model generation errors
    """
    def __init__(
        self, 
        message: str, 
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MAJOR,
        original_exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        """
        Initialize ModelError
        
        Args:
            message: Error message
            error_type: Type of error
            severity: Error severity level
            original_exception: Original exception that caused this error
            context: Additional context information
            suggestions: Suggested fixes or actions
        """
        self.error_type = error_type
        self.severity = severity
        self.original_exception = original_exception
        self.context = context or {}
        self.suggestions = suggestions or []
        self.recovery_attempts = []
        
        # Format the message with additional information
        formatted_message = f"{error_type.value.upper()} ERROR ({severity.value}): {message}"
        super().__init__(formatted_message)
    
    def add_recovery_attempt(self, strategy: str, success: bool, result: Any = None):
        """
        Add a recovery attempt to the error
        
        Args:
            strategy: Name of the recovery strategy attempted
            success: Whether the recovery was successful
            result: Result of the recovery attempt
        """
        self.recovery_attempts.append({
            "strategy": strategy,
            "success": success,
            "result": result
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary representation
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "message": str(self),
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "context": self.context,
            "suggestions": self.suggestions,
            "recovery_attempts": self.recovery_attempts,
            "traceback": traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            ) if self.original_exception else None
        }
    
    def to_json(self) -> str:
        """
        Convert error to JSON representation
        
        Returns:
            JSON representation of the error
        """
        return json.dumps(self.to_dict(), indent=2)


class ErrorClassifier:
    """
    Classifies errors based on their characteristics
    """
    
    @staticmethod
    def classify_exception(
        exception: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> ModelError:
        """
        Classify an exception into a ModelError
        
        Args:
            exception: The exception to classify
            context: Additional context information
        
        Returns:
            Classified ModelError
        """
        error_message = str(exception)
        context = context or {}
        
        # Classify parsing errors
        if "Error parsing LLM output" in error_message:
            return ModelError(
                message=error_message,
                error_type=ErrorType.PARSING,
                severity=ErrorSeverity.MAJOR,
                original_exception=exception,
                context=context,
                suggestions=[
                    "Check LLM output format requirements",
                    "Adjust prompts to guide LLM toward correct format",
                    "Implement more flexible parsing logic"
                ]
            )
        
        # Classify POWL code execution errors
        if "Error executing POWL code" in error_message:
            severity = ErrorSeverity.CRITICAL if "has no attribute" in error_message else ErrorSeverity.MAJOR
            return ModelError(
                message=error_message,
                error_type=ErrorType.EXECUTION,
                severity=severity,
                original_exception=exception,
                context=context,
                suggestions=[
                    "Check for type errors in the generated code",
                    "Ensure proper variable initialization",
                    "Validate code structure before execution"
                ]
            )
        
        # Classify validation errors
        if "Error validating POWL model" in error_message:
            severity = ErrorSeverity.MAJOR
            suggestions = []
            
            if "Cannot create an xor of less than 2 submodels" in error_message:
                suggestions = [
                    "Ensure XOR operations have at least 2 submodels",
                    "Check model structure for incomplete branches",
                    "Add default paths for decision points"
                ]
            
            return ModelError(
                message=error_message,
                error_type=ErrorType.VALIDATION,
                severity=severity,
                original_exception=exception,
                context=context,
                suggestions=suggestions or [
                    "Validate model structure before finalization",
                    "Check for common structural issues",
                    "Simplify complex model sections"
                ]
            )
        
        # Classify syntax errors
        if isinstance(exception, SyntaxError):
            return ModelError(
                message=f"Syntax error at line {getattr(exception, 'lineno', '?')}: {error_message}",
                error_type=ErrorType.EXECUTION,
                severity=ErrorSeverity.MAJOR,
                original_exception=exception,
                context=context,
                suggestions=[
                    "Check for missing parentheses or brackets",
                    "Ensure proper indentation",
                    "Verify correct parameter names"
                ]
            )
        
        # Classify type errors
        if isinstance(exception, TypeError):
            return ModelError(
                message=f"Type error: {error_message}",
                error_type=ErrorType.EXECUTION,
                severity=ErrorSeverity.MAJOR,
                original_exception=exception,
                context=context,
                suggestions=[
                    "Check variable types and conversions",
                    "Ensure proper method arguments",
                    "Verify object attributes exist before access"
                ]
            )
        
        # Default classification
        return ModelError(
            message=error_message,
            error_type=ErrorType.UNKNOWN,
            severity=ErrorSeverity.MAJOR,
            original_exception=exception,
            context=context
        )


class RecoveryStrategy:
    """
    Base class for error recovery strategies
    """
    
    def __init__(self, name: str):
        """
        Initialize recovery strategy
        
        Args:
            name: Name of the recovery strategy
        """
        self.name = name
    
    def can_handle(self, error: ModelError) -> bool:
        """
        Check if this strategy can handle the given error
        
        Args:
            error: The error to check
        
        Returns:
            True if this strategy can handle the error, False otherwise
        """
        raise NotImplementedError("Subclasses must implement can_handle")
    
    def recover(self, error: ModelError, **kwargs) -> Tuple[bool, Any]:
        """
        Attempt to recover from the error
        
        Args:
            error: The error to recover from
            **kwargs: Additional arguments for recovery
        
        Returns:
            Tuple of (success, result)
        """
        raise NotImplementedError("Subclasses must implement recover")


class LLMOutputParsingRecovery(RecoveryStrategy):
    """
    Recovery strategy for LLM output parsing errors
    """
    
    def __init__(self):
        """Initialize LLM output parsing recovery strategy"""
        super().__init__("LLM Output Parsing Recovery")
    
    def can_handle(self, error: ModelError) -> bool:
        """Check if this strategy can handle the error"""
        return error.error_type == ErrorType.PARSING
    
    def recover(self, error: ModelError, **kwargs) -> Tuple[bool, Any]:
        """
        Attempt to recover from LLM output parsing error
        
        Args:
            error: The parsing error
            **kwargs: Additional arguments including:
                - raw_output: The raw LLM output to parse
                - expected_format: The expected format description
        
        Returns:
            Tuple of (success, parsed_result or error_message)
        """
        raw_output = kwargs.get("raw_output", "")
        if not raw_output:
            return False, "No raw output provided for recovery"
        
        # Try multiple parsing strategies with decreasing strictness
        
        # 1. Try to extract Python code blocks
        try:
            python_code = self._extract_python_code(raw_output)
            if python_code:
                return True, python_code
        except Exception as e:
            logger.debug(f"Python code extraction failed: {str(e)}")
        
        # 2. Try to extract structured content based on keywords
        try:
            structured_content = self._extract_structured_content(raw_output)
            if structured_content:
                return True, structured_content
        except Exception as e:
            logger.debug(f"Structured content extraction failed: {str(e)}")
        
        # 3. Try to salvage any usable content
        try:
            salvaged_content = self._salvage_content(raw_output)
            if salvaged_content:
                return True, salvaged_content
        except Exception as e:
            logger.debug(f"Content salvaging failed: {str(e)}")
        
        return False, "All parsing recovery strategies failed"
    
    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from text"""
        # Try markdown code blocks
        python_code_pattern = r"```python(.*?)```"
        matches = re.findall(python_code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        # Try code without markdown formatting
        code_pattern = r"from promoagent_plus\.core\.model_generator import ModelGenerator(.*?)final_model ="
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            code = "from promoagent_plus.core.model_generator import ModelGenerator" + matches[0]
            # Find the final_model assignment
            final_line_pattern = r"final_model = .*"
            final_line_match = re.search(final_line_pattern, text)
            if final_line_match:
                code += final_line_match.group(0)
                return code
        
        return None
    
    def _extract_structured_content(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured content based on keywords"""
        # Look for common section headers
        sections = {}
        
        # Try to find "Final Answer:" section
        final_answer_match = re.search(r"Final Answer:(.*?)(?:\n\n|$)", text, re.DOTALL)
        if final_answer_match:
            sections["final_answer"] = final_answer_match.group(1).strip()
        
        # Try to find "Thought:" section
        thought_match = re.search(r"Thought:(.*?)(?:Action:|Final Answer:|$)", text, re.DOTALL)
        if thought_match:
            sections["thought"] = thought_match.group(1).strip()
        
        # Try to find "Action:" section
        action_match = re.search(r"Action:(.*?)(?:Action Input:|Final Answer:|$)", text, re.DOTALL)
        if action_match:
            sections["action"] = action_match.group(1).strip()
        
        # Try to find "Action Input:" section
        action_input_match = re.search(r"Action Input:(.*?)(?:Observation:|Final Answer:|$)", text, re.DOTALL)
        if action_input_match:
            sections["action_input"] = action_input_match.group(1).strip()
        
        return sections if sections else None
    
    def _salvage_content(self, text: str) -> Optional[str]:
        """Salvage any usable content from text"""
        # Look for code-like content
        code_indicators = ["import", "def ", "class ", "gen.", "ModelGenerator", "final_model"]
        lines = text.split("\n")
        code_lines = []
        
        in_code_block = False
        for line in lines:
            # Check if this line indicates code
            is_code_line = any(indicator in line for indicator in code_indicators)
            
            # If we're in a code block or this is a code line, add it
            if in_code_block or is_code_line:
                code_lines.append(line)
                in_code_block = True
            
            # End of code block if we hit an empty line after code
            if in_code_block and not line.strip():
                in_code_block = False
        
        return "\n".join(code_lines) if code_lines else None


class POWLCodeExecutionRecovery(RecoveryStrategy):
    """
    Recovery strategy for POWL code execution errors
    """
    
    def __init__(self):
        """Initialize POWL code execution recovery strategy"""
        super().__init__("POWL Code Execution Recovery")
    
    def can_handle(self, error: ModelError) -> bool:
        """Check if this strategy can handle the error"""
        return error.error_type == ErrorType.EXECUTION
    
    def recover(self, error: ModelError, **kwargs) -> Tuple[bool, Any]:
        """
        Attempt to recover from POWL code execution error
        
        Args:
            error: The execution error
            **kwargs: Additional arguments including:
                - code: The POWL code that failed to execute
        
        Returns:
            Tuple of (success, fixed_code or error_message)
        """
        code = kwargs.get("code", "")
        if not code:
            return False, "No code provided for recovery"
        
        # Apply multiple fixes and return the first successful one
        
        # 1. Fix indentation issues
        try:
            fixed_code = self._fix_indentation(code)
            return True, fixed_code
        except Exception as e:
            logger.debug(f"Indentation fixing failed: {str(e)}")
        
        # 2. Fix missing imports
        try:
            fixed_code = self._fix_missing_imports(code)
            return True, fixed_code
        except Exception as e:
            logger.debug(f"Import fixing failed: {str(e)}")
        
        # 3. Fix variable references
        try:
            fixed_code = self._fix_variable_references(code)
            return True, fixed_code
        except Exception as e:
            logger.debug(f"Variable reference fixing failed: {str(e)}")
        
        # 4. Fix type errors
        try:
            fixed_code = self._fix_type_errors(code, error)
            return True, fixed_code
        except Exception as e:
            logger.debug(f"Type error fixing failed: {str(e)}")
        
        return False, "All code execution recovery strategies failed"
    
    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues in code"""
        lines = code.split("\n")
        fixed_lines = []
        
        for line in lines:
            # Remove any leading whitespace
            stripped = line.lstrip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append("")
                continue
            
            # Determine correct indentation level based on context
            # This is a simplified approach - a real implementation would be more sophisticated
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
                fixed_lines.append("    " + stripped)
        
        return "\n".join(fixed_lines)
    
    def _fix_missing_imports(self, code: str) -> str:
        """Fix missing imports in code"""
        required_imports = [
            "from promoagent_plus.core.model_generator import ModelGenerator",
            "from pm4py.objects.powl.obj import POWL, Transition, SilentTransition, StrictPartialOrder, OperatorPOWL, Operator"
        ]
        
        # Check if imports are already present
        for imp in required_imports:
            if imp not in code:
                # Add missing import at the beginning
                code = imp + "\n" + code
        
        # Ensure ModelGenerator initialization
        if "gen = ModelGenerator()" not in code:
            # Find the right place to add it (after imports)
            lines = code.split("\n")
            import_end_idx = 0
            for i, line in enumerate(lines):
                if line.startswith(("import ", "from ")) or not line.strip():
                    import_end_idx = i
                else:
                    break
            
            # Insert after imports
            lines.insert(import_end_idx + 1, "gen = ModelGenerator()")
            code = "\n".join(lines)
        
        return code
    
    def _fix_variable_references(self, code: str) -> str:
        """Fix variable reference issues in code"""
        lines = code.split("\n")
        var_pattern = r"var_\d+"
        
        # Find all variable definitions
        defined_vars = set()
        for line in lines:
            if "=" in line:
                var_name = line.split("=")[0].strip()
                if re.match(var_pattern, var_name):
                    defined_vars.add(var_name)
        
        # Check for undefined variable references
        for i, line in enumerate(lines):
            if "=" in line and "gen." in line:
                right_side = line.split("=")[1]
                for var_ref in re.findall(var_pattern, right_side):
                    if var_ref not in defined_vars:
                        # Replace undefined variable with a silent transition
                        lines[i] = lines[i].replace(var_ref, "gen.silent_transition()")
        
        return "\n".join(lines)
    
    def _fix_type_errors(self, code: str, error: ModelError) -> str:
        """Fix type errors in code based on error message"""
        error_msg = str(error.original_exception) if error.original_exception else str(error)
        
        # Handle 'CrewOutput' object has no attribute 'split' error
        if "'CrewOutput' object has no attribute 'split'" in error_msg:
            # Find the line that's trying to split a CrewOutput
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if ".split" in line:
                    # Replace with str() conversion
                    lines[i] = lines[i].replace(".split", ".__str__().split")
            return "\n".join(lines)
        
        # Handle other common type errors
        # This would be expanded based on observed error patterns
        
        return code


class POWLModelValidationRecovery(RecoveryStrategy):
    """
    Recovery strategy for POWL model validation errors
    """
    
    def __init__(self):
        """Initialize POWL model validation recovery strategy"""
        super().__init__("POWL Model Validation Recovery")
    
    def can_handle(self, error: ModelError) -> bool:
        """Check if this strategy can handle the error"""
        return error.error_type == ErrorType.VALIDATION
    
    def recover(self, error: ModelError, **kwargs) -> Tuple[bool, Any]:
        """
        Attempt to recover from POWL model validation error
        
        Args:
            error: The validation error
            **kwargs: Additional arguments including:
                - code: The POWL code that failed validation
                - model: The POWL model object (if available)
        
        Returns:
            Tuple of (success, fixed_code or error_message)
        """
        code = kwargs.get("code", "")
        if not code:
            return False, "No code provided for recovery"
        
        error_msg = str(error.original_exception) if error.original_exception else str(error)
        
        # Handle specific validation errors
        
        # 1. Fix "Cannot create an xor of less than 2 submodels" error
        if "Cannot create an xor of less than 2 submodels" in error_msg:
            try:
                fixed_code = self._fix_xor_submodels(code)
                return True, fixed_code
            except Exception as e:
                logger.debug(f"XOR submodel fixing failed: {str(e)}")
        
        # 2. Fix partial order issues
        if "partial order" in error_msg.lower():
            try:
                fixed_code = self._fix_partial_order(code)
                return True, fixed_code
            except Exception as e:
                logger.debug(f"Partial order fixing failed: {str(e)}")
        
        # 3. General model structure fixes
        try:
            fixed_code = self._fix_model_structure(code)
            return True, fixed_code
        except Exception as e:
            logger.debug(f"Model structure fixing failed: {str(e)}")
        
        return False, "All model validation recovery strategies failed"
    
    def _fix_xor_submodels(self, code: str) -> str:
        """Fix XOR with less than 2 submodels"""
        lines = code.split("\n")
        
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
        
        return "\n".join(lines)
    
    def _fix_partial_order(self, code: str) -> str:
        """Fix partial order issues"""
        # This is a placeholder for partial order fixes
        # A real implementation would analyze and fix specific partial order issues
        return code
    
    def _fix_model_structure(self, code: str) -> str:
        """Fix general model structure issues"""
        lines = code.split("\n")
        
        # Ensure final_model is assigned
        has_final_model = any("final_model =" in line for line in lines)
        if not has_final_model:
            # Find the last variable assignment
            last_var = None
            for line in lines:
                if "=" in line and "var_" in line.split("=")[0]:
                    last_var = line.split("=")[0].strip()
            
            if last_var:
                lines.append(f"final_model = {last_var}")
        
        return "\n".join(lines)


class ErrorHandler:
    """
    Central error handling system for ProMoAgent+
    """
    
    def __init__(self):
        """Initialize ErrorHandler with recovery strategies"""
        self.recovery_strategies = [
            LLMOutputParsingRecovery(),
            POWLCodeExecutionRecovery(),
            POWLModelValidationRecovery()
        ]
        self.error_history = []
    
    def handle_exception(
        self, 
        exception: Exception, 
        context: Optional[Dict[str, Any]] = None,
        **recovery_kwargs
    ) -> Tuple[bool, Any, Optional[ModelError]]:
        """
        Handle an exception with appropriate recovery strategies
        
        Args:
            exception: The exception to handle
            context: Additional context information
            **recovery_kwargs: Additional arguments for recovery strategies
        
        Returns:
            Tuple of (recovered, result, error)
            - recovered: Whether recovery was successful
            - result: Recovery result if successful, None otherwise
            - error: ModelError object for logging and reporting
        """
        # Classify the exception
        error = ErrorClassifier.classify_exception(exception, context)
        
        # Log the error
        logger.error(f"Handling error: {str(error)}")
        self.error_history.append(error)
        
        # Try recovery strategies
        for strategy in self.recovery_strategies:
            if strategy.can_handle(error):
                logger.info(f"Attempting recovery with strategy: {strategy.name}")
                try:
                    success, result = strategy.recover(error, **recovery_kwargs)
                    error.add_recovery_attempt(strategy.name, success, str(result) if success else result)
                    
                    if success:
                        logger.info(f"Recovery successful with strategy: {strategy.name}")
                        return True, result, error
                except Exception as e:
                    logger.error(f"Recovery strategy {strategy.name} failed: {str(e)}")
                    error.add_recovery_attempt(strategy.name, False, str(e))
        
        # All recovery strategies failed
        logger.error("All recovery strategies failed")
        return False, None, error
    
    def get_error_history(self) -> List[ModelError]:
        """
        Get the error history
        
        Returns:
            List of ModelError objects
        """
        return self.error_history
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors and recovery attempts
        
        Returns:
            Dictionary with error statistics
        """
        stats = {
            "total_errors": len(self.error_history),
            "by_type": {},
            "by_severity": {},
            "recovery_success_rate": 0,
            "most_common_errors": []
        }
        
        # Count errors by type and severity
        for error in self.error_history:
            # By type
            error_type = error.error_type.value
            if error_type not in stats["by_type"]:
                stats["by_type"][error_type] = 0
            stats["by_type"][error_type] += 1
            
            # By severity
            severity = error.severity.value
            if severity not in stats["by_severity"]:
                stats["by_severity"][severity] = 0
            stats["by_severity"][severity] += 1
        
        # Calculate recovery success rate
        successful_recoveries = sum(
            1 for error in self.error_history
            if any(attempt["success"] for attempt in error.recovery_attempts)
        )
        
        if stats["total_errors"] > 0:
            stats["recovery_success_rate"] = successful_recoveries / stats["total_errors"]
        
        # Find most common error messages
        error_messages = {}
        for error in self.error_history:
            msg = str(error.original_exception) if error.original_exception else str(error)
            if msg not in error_messages:
                error_messages[msg] = 0
            error_messages[msg] += 1
        
        # Sort by frequency
        sorted_errors = sorted(error_messages.items(), key=lambda x: x[1], reverse=True)
        stats["most_common_errors"] = [{"message": msg, "count": count} for msg, count in sorted_errors[:5]]
        
        return stats


# Global error handler instance
error_handler = ErrorHandler()

def safe_execute(
    func: Callable, 
    error_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[bool, Any, Optional[ModelError]]:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        error_context: Context information for error handling
        **kwargs: Arguments to pass to the function and recovery strategies
    
    Returns:
        Tuple of (success, result, error)
    """
    try:
        result = func(**kwargs)
        return True, result, None
    except Exception as e:
        return error_handler.handle_exception(e, error_context, **kwargs)
