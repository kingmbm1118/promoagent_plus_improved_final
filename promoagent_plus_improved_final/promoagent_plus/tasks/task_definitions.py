"""
Task definitions for ProMoAgent+ agents
"""

from crewai import Task, Agent
from promoagent_plus.utils.validation_utils import get_error_specific_suggestions


def create_analysis_task(agent: Agent, process_description: str) -> Task:
    """
    Create a task for analyzing a process description
    
    Args:
        agent: The process analyzer agent
        process_description: The process description text
        
    Returns:
        Analysis task
    """
    return Task(
        description=f"""
        Analyze the following process description in detail:
        
        {process_description}
        
        Your goal is to:
        1. Identify all activities in the process
        2. Determine actors/roles involved
        3. Identify control flow patterns (sequences, choices, loops, concurrency)
        4. Note any conditions that affect the process flow
        5. Create a structured summary of the process
        
        Provide your analysis in a structured format that includes:
        - List of activities with descriptions
        - Control flow relationships between activities
        - Decision points and their conditions
        - Any loops or repetitions in the process
        """,
        agent=agent,
        expected_output="A structured analysis of the process with activities, control flow, and conditions"
    )


def create_modeling_task_with_feedback(agent: Agent, analysis_result: str, error_message: str, previous_code: str) -> Task:
    """
    Create a task for creating a POWL model with feedback from previous errors
    
    Args:
        agent: The POWL modeler agent
        analysis_result: The result of the process analysis
        error_message: The error message from previous attempt
        previous_code: The code from previous attempt
        
    Returns:
        Modeling task with feedback
    """
    return Task(
        description=f"""
        Based on the following process analysis, create a POWL (Partial Order Workflow Language) model:
        
        {analysis_result}
        
        Your previous attempt resulted in the following error:
        ```
        {error_message}
        ```
        
        Your previous code was:
        ```python
        {previous_code}
        ```
        
        Please correct the issues and create a valid POWL model. 
        
        Here are some specific suggestions based on the error:
        {get_error_specific_suggestions(error_message)}
        
        Remember these important guidelines:
        - Ensure each submodel is used uniquely
        - Use proper nesting of operators
        - Don't create partial orders as children of other partial orders
        - Model XOR between complete alternative paths, not just decision points
        - Fix indentation issues - make sure there are no leading spaces in your code
        - Verify function parameters match the expected API
        
        Your output should be complete, executable Python code using the ModelGenerator class.
        The final POWL model should be saved in a variable named 'final_model'.
        """,
        agent=agent,
        expected_output="Python code that creates a POWL model using ModelGenerator"
    )

def create_modeling_task_with_intervention(agent: Agent, analysis_result: str, intervention_suggestions: str) -> Task:
    """
    Create a task for creating a POWL model with intervention suggestions
    
    Args:
        agent: The POWL modeler agent
        analysis_result: The result of the process analysis
        intervention_suggestions: Suggestions to get unstuck
        
    Returns:
        Modeling task with intervention
    """
    return Task(
        description=f"""
        Based on the following process analysis, create a POWL (Partial Order Workflow Language) model:
        
        {analysis_result}
        
        You appear to be stuck in a loop. Please try a COMPLETELY DIFFERENT approach.
        
        Consider these suggestions:
        {intervention_suggestions}
        
        IMPORTANT: Do not repeat your previous approach. Take a step back and rethink the problem from scratch.
        
        Start with the absolute simplest model that could work, then build up complexity only after validating each step.
        
        Your output should be complete, executable Python code using the ModelGenerator class.
        The final POWL model should be saved in a variable named 'final_model'.
        """,
        agent=agent,
        expected_output="Python code that creates a POWL model using ModelGenerator"
    )

def create_modeling_task_with_feedback(agent: Agent, analysis_result: str, error_message: str, previous_code: str) -> Task:
    """
    Create a task for creating a POWL model with feedback from previous errors
    
    Args:
        agent: The POWL modeler agent
        analysis_result: The result of the process analysis
        error_message: The error message from previous attempt
        previous_code: The code from previous attempt
        
    Returns:
        Modeling task with feedback
    """
    return Task(
        description=f"""
        Based on the following process analysis, create a POWL (Partial Order Workflow Language) model:
        
        {analysis_result}
        
        Your previous attempt resulted in the following error:
        ```
        {error_message}
        ```
        
        Your previous code was:
        ```python
        {previous_code}
        ```
        
        Please correct the issues and create a valid POWL model. 
        
        Here are some specific suggestions based on the error:
        {get_error_specific_suggestions(error_message)}
        
        Remember these important guidelines:
        - Ensure each submodel is used uniquely
        - Use proper nesting of operators
        - Don't create partial orders as children of other partial orders
        - Model XOR between complete alternative paths, not just decision points
        - Fix indentation issues - make sure there are no leading spaces in your code
        - Verify function parameters match the expected API
        
        Your output should be complete, executable Python code using the ModelGenerator class.
        The final POWL model should be saved in a variable named 'final_model'.
        """,
        agent=agent,
        expected_output="Python code that creates a POWL model using ModelGenerator"
    )

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

def create_modeling_task(agent: Agent, analysis_result: str) -> Task:
    """
    Create a task for creating a POWL model based on analysis
    
    Args:
        agent: The POWL modeler agent
        analysis_result: The result of the process analysis
        
    Returns:
        Modeling task
    """
    return Task(
        description=f"""
        Based on the following process analysis, create a POWL (Partial Order Workflow Language) model:
        
        {analysis_result}
        
        Your task is to:
        1. Identify the base activities from the analysis
        2. Model exclusive choices using the xor() operator
        3. Model loops using the loop() operator
        4. Model concurrent activities with dependencies using partial_order()
        
        Remember these important guidelines:
        - Ensure each submodel is used uniquely
        - Use proper nesting of operators
        - Don't create partial orders as children of other partial orders
        - Model XOR between complete alternative paths, not just decision points
        
        If you encounter errors:
        - Read the error message carefully and address the specific issue
        - Start with a simpler model and gradually add complexity
        - Focus on fixing one issue at a time before moving on
        - If stuck with the same error, try a completely different approach
        - Pay close attention to function signatures and parameters
        
        Your output should be complete, executable Python code using the ModelGenerator class.
        The final POWL model should be saved in a variable named 'final_model'.
        
        Example structure:
        ```python
        from promoagent_plus.core.model_generator import ModelGenerator
        
        gen = ModelGenerator()
        # Create activities and combine them
        activity_a = gen.activity('Activity A')
        ...
        final_model = gen.partial_order(dependencies=[(activity_a, ...)])
        ```
        """,
        agent=agent,
        expected_output="Python code that creates a POWL model using ModelGenerator"
    )

def create_review_task(agent: Agent, powl_code: str, original_description: str) -> Task:
    """
    Create a task for reviewing a POWL model
    
    Args:
        agent: The model reviewer agent
        powl_code: The Python code for the POWL model
        original_description: The original process description
        
    Returns:
        Review task
    """
    return Task(
        description=f"""
        Review the following POWL model code against the original process description:
        
        Original Process Description:
        {original_description}
        
        POWL Model Code:
        ```python
        {powl_code}
        ```
        
        Your task is to:
        1. Verify that the model correctly represents all activities in the description
        2. Check that control flow (sequence, choice, loops, concurrency) is accurately modeled
        3. Validate that all conditions and decision points are properly implemented
        4. Identify any missing elements or logical errors
        5. Suggest specific improvements
        
        Provide a detailed review with specific feedback on:
        - Correctness: Does the model match the description?
        - Completeness: Are all elements of the process included?
        - Structure: Is the model well-structured and using appropriate POWL constructs?
        - Improvements: What specific changes would enhance the model?
        """,
        agent=agent,
        expected_output="A detailed review of the POWL model with specific feedback and improvement suggestions"
    )

def create_improvement_task(agent: Agent, review_result: str, original_powl_code: str) -> Task:
    """
    Create a task for improving a POWL model based on review
    
    Args:
        agent: The POWL modeler agent
        review_result: The result of the model review
        original_powl_code: The original Python code for the POWL model
        
    Returns:
        Improvement task
    """
    return Task(
        description=f"""
        Improve the POWL model based on the review feedback:
        
        Review Feedback:
        {review_result}
        
        Original POWL Model Code:
        ```python
        {original_powl_code}
        ```
        
        Your task is to:
        1. Address all issues identified in the review
        2. Implement the suggested improvements
        3. Create an improved version of the POWL model code
        
        Your output should be complete, executable Python code using the ModelGenerator class.
        The final improved POWL model should be saved in a variable named 'final_model'.
        """,
        agent=agent,
        expected_output="Improved Python code that creates a better POWL model"
    )

def create_translation_task(agent: Agent, powl_code: str, target_format: str) -> Task:
    """
    Create a task for translating a POWL model to another format
    
    Args:
        agent: The model translator agent
        powl_code: The Python code for the POWL model
        target_format: The target format (e.g., "BPMN", "Petri Net")
        
    Returns:
        Translation task
    """
    return Task(
        description=f"""
        Translate the following POWL model to {target_format} format:
        
        POWL Model Code:
        ```python
        {powl_code}
        ```
        
        Your task is to:
        1. Convert the POWL model to {target_format} representation
        2. Ensure the semantic equivalence of the translation
        3. Explain any notation-specific adaptations made during translation
        
        Your output should include:
        - Confirmation that the translation was successful
        - Any important notes about the translation process
        - Information about any semantic differences between the original and translated models
        """,
        agent=agent,
        expected_output=f"Confirmation of successful translation to {target_format} with notes on the process"
    )

def create_user_feedback_task(agent: Agent, powl_code: str, feedback: str, original_description: str) -> Task:
    """
    Create a task for incorporating user feedback into a POWL model
    
    Args:
        agent: The POWL modeler agent
        powl_code: The Python code for the POWL model
        feedback: The user feedback
        original_description: The original process description
        
    Returns:
        Feedback incorporation task
    """
    return Task(
        description=f"""
        Update the POWL model based on the user feedback:
        
        Original Process Description:
        {original_description}
        
        Current POWL Model Code:
        ```python
        {powl_code}
        ```
        
        User Feedback:
        {feedback}
        
        Your task is to:
        1. Understand the user's feedback and how it relates to the process
        2. Modify the POWL model to address the feedback
        3. Ensure the modified model still accurately represents the original process where not affected by the feedback
        4. Create an updated version of the POWL model code
        
        Your output should be complete, executable Python code using the ModelGenerator class.
        The final updated POWL model should be saved in a variable named 'final_model'.
        """,
        agent=agent,
        expected_output="Updated Python code that incorporates the user feedback into the POWL model"
    )