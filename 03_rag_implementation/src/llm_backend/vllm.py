from openai import OpenAI

class VLLMBackend():
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def send_request(self, model, messages, temperature, max_tokens, thinking="none"):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": thinking != "none"},
            }
        )