"""
Constants used throughout the ProMoAgent+ system
"""

from enum import Enum

class AIProviders(Enum):
    """Supported AI providers for the LLM agents"""
    GOOGLE = "Google"
    OPENAI = "OpenAI"
    DEEPSEEK = "DeepSeek"
    ANTHROPIC = "Anthropic"
    DEEPINFRA = "Deepinfra"
    MISTRAL_AI = "Mistral AI"

# Configuration for default models
AI_MODEL_DEFAULTS = {
    AIProviders.GOOGLE.value: 'gemini-2.5-pro-exp-03-25',
    AIProviders.OPENAI.value: 'gpt-4',
    AIProviders.DEEPSEEK.value: 'deepseek-reasoner',
    AIProviders.ANTHROPIC.value: 'claude-3-5-sonnet-latest',
    AIProviders.DEEPINFRA.value: 'meta-llama/Llama-3.2-90B-Vision-Instruct',
    AIProviders.MISTRAL_AI.value: 'mistral-large-latest'
}

DEFAULT_AI_PROVIDER = AIProviders.ANTHROPIC.value

class InputType(Enum):
    """Types of input for process model generation"""
    TEXT = "Text"
    MODEL = "Model"
    DATA = "Data"

class ViewType(Enum):
    """Types of process model visualization"""
    BPMN = "BPMN"
    POWL = "POWL"
    PETRI = "Petri Net"