from .vllm import VLLMBackend
from .openai import OpenAIBackend

def load_llm_backend(base_url, api_key):
    if 'api.openai.com' in base_url:
        return OpenAIBackend(base_url, api_key)
    return VLLMBackend(base_url, api_key)