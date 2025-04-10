"""
Utility functions for working with POWL models
"""

import re
from pm4py.objects.powl.obj import POWL, Transition, SilentTransition, StrictPartialOrder, OperatorPOWL, Operator

def powl_to_code(powl_obj: POWL) -> str:
    """
    Translates a POWL object from pm4py into code using ModelGenerator.
    
    Args:
        powl_obj: The POWL object to translate.
    
    Returns:
        A string containing the Python code that constructs the equivalent POWL model using ModelGenerator.
    """
    import_statement = 'from promoagent_plus.core.model_generator import ModelGenerator'
    code_lines = [import_statement, 'gen = ModelGenerator()']

    var_counter = [0]

    def get_new_var_name():
        var_name = f"var_{var_counter[0]}"
        var_counter[0] += 1
        return var_name

    def process_powl(powl):
        if isinstance(powl, Transition):
            var_name = get_new_var_name()
            if isinstance(powl, SilentTransition):
                code_lines.append(f"{var_name} = gen.silent_transition()")
            else:
                label = powl.label
                code_lines.append(f"{var_name} = gen.activity('{label}')")
            return var_name

        elif isinstance(powl, OperatorPOWL):
            operator = powl.operator
            children = powl.children
            child_vars = [process_powl(child) for child in children]
            var_name = get_new_var_name()
            if operator == Operator.XOR:
                child_vars_str = ', '.join(child_vars)
                code_lines.append(f"{var_name} = gen.xor({child_vars_str})")
            elif operator == Operator.LOOP:
                if len(child_vars) != 2:
                    raise Exception("A loop of invalid size! This should not be possible!")
                do_var = child_vars[0]
                redo_var = child_vars[1]
                code_lines.append(f"{var_name} = gen.loop(do={do_var}, redo={redo_var})")
            else:
                raise Exception("Unknown operator! This should not be possible!")
            return var_name

        elif isinstance(powl, StrictPartialOrder):
            nodes = powl.get_children()
            order = powl.order.get_transitive_reduction()
            node_var_map = {node: process_powl(node) for node in nodes}
            dependencies = []
            nodes_in_edges = set()
            for source in nodes:
                for target in nodes:
                    source_var = node_var_map[source]
                    target_var = node_var_map[target]
                    if order.is_edge(source, target):
                        dependencies.append(f"({source_var}, {target_var})")
                        nodes_in_edges.update([source, target])

            # Include nodes not in any edge as singleton tuples
            for node in nodes:
                if node not in nodes_in_edges:
                    var = node_var_map[node]
                    dependencies.append(f"({var},)")

            dep_str = ', '.join(dependencies)
            var_name = get_new_var_name()
            code_lines.append(f"{var_name} = gen.partial_order(dependencies=[{dep_str}])")
            return var_name

        else:
            raise Exception("Unknown POWL object! This should not be possible!")

    final_var = process_powl(powl_obj)
    code_lines.append(f"final_model = {final_var}")

    return '\n'.join(code_lines)

def extract_powl_code(response_text: str) -> str:
    """
    Extract Python code for POWL model from LLM response
    
    Args:
        response_text: The LLM response text
    
    Returns:
        Extracted Python code
    """
    python_code_pattern = r"```python(.*?)```"
    matches = re.findall(python_code_pattern, response_text, re.DOTALL)
    
    if matches:
        python_snippet = matches[-1].strip()
        return python_snippet
    else:
        # Try without markdown formatting as fallback
        code_pattern = r"from promoagent_plus\.core\.model_generator import ModelGenerator(.*?)final_model ="
        matches = re.findall(code_pattern, response_text, re.DOTALL, re.MULTILINE)
        if matches:
            code = "from promoagent_plus.core.model_generator import ModelGenerator" + matches[0]
            # Find the final_model assignment
            final_line_pattern = r"final_model = .*"
            final_line_match = re.search(final_line_pattern, response_text)
            if final_line_match:
                code += final_line_match.group(0)
                return code
        
        raise Exception("No Python code snippet found in the response!")

def execute_powl_code(code: str) -> POWL:
    """
    Execute POWL code and get the resulting model
    
    Args:
        code: The Python code for creating a POWL model
    
    Returns:
        The POWL model
    """
    from promoagent_plus.core.model_generator import ModelGenerator
    
    try:
        # Clean up code indentation
        cleaned_code = "\n".join([line.lstrip() for line in code.split("\n")])
        
        local_vars = {"ModelGenerator": ModelGenerator}
        exec(f"from pm4py.objects.powl.obj import POWL, Transition, SilentTransition, StrictPartialOrder, OperatorPOWL, Operator\n{cleaned_code}", globals(), local_vars)
        
        if "final_model" not in local_vars:
            # Try alternative variable names if final_model isn't found
            potential_models = [v for k, v in local_vars.items() if isinstance(v, POWL)]
            if potential_models:
                return potential_models[0]
            raise ValueError("Variable 'final_model' not found in the code!")
        
        model = local_vars["final_model"]
        if not isinstance(model, POWL):
            raise TypeError(f"The final_model is not a POWL model. Got {type(model)}")
        
        return model
    except SyntaxError as e:
        # Try to provide helpful context about the syntax error
        line_no = getattr(e, 'lineno', None)
        if line_no:
            lines = code.split('\n')
            context = lines[max(0, line_no-2):min(len(lines), line_no+1)]
            context_str = '\n'.join(context)
            raise Exception(f"Syntax error at line {line_no}: {str(e)}\nContext:\n{context_str}")
        raise Exception(f"Syntax error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error executing POWL code: {str(e)}")

def validate_powl_model(model: POWL) -> bool:
    """
    Validate a POWL model
    
    Args:
        model: The POWL model to validate
    
    Returns:
        True if valid, raises exception otherwise
    """
    # Validate partial orders
    def validate_partial_orders(powl):
        if isinstance(powl, StrictPartialOrder):
            # Check irreflexivity and add transitive edges if needed
            if not powl.order.is_irreflexive():
                raise Exception("The irreflexivity of the partial order is violated!")
            
            if not powl.order.is_transitive():
                powl.order.add_transitive_edges()
                if not powl.order.is_irreflexive():
                    raise Exception("The transitive closure of the provided relation violates irreflexivity!")
        
        # Recursively validate children
        if hasattr(powl, 'children'):
            for child in powl.children:
                validate_partial_orders(child)
    
    validate_partial_orders(model)
    return True