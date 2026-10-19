from openai import OpenAI ,OpenAIError
from app.config import settings
from app.schemas import AssistantStructuredResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMService:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY  
        )
        #fallback
        self.fallback_client = OpenAI(
            api_key=getattr(settings, "OPENAI_API_KEY", "fallback_key")
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True
    )
    def _call_primary_model(self, messages: list):
        return self.client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=settings.TEMPERATURE,
            response_format={"type": "json_object"}
        )
        
    def generate_response(self, messages: list):
        try:
            # Attempt primary client with exponential retries
            response = self._call_primary_model(messages)
            return response.choices[0].message.content, "primary"
        except Exception as primary_error:
            # Fallback to secondary model if primary repeatedly fails
            try:
                fallback_resp = self.fallback_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=settings.TEMPERATURE
                )
                return fallback_resp.choices[0].message.content, "fallback"
            except Exception as fallback_error:
                raise RuntimeError(f"Both Primary and Fallback failed: {fallback_error}")   
    def generate_with_rag(self, prompt: str, retrieved_context: list[str]):
        context_str = "\n".join(retrieved_context)
        formatted_prompt = f"Context:\n{context_str}\n\nUser Question: {prompt}"
        
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            messages=[
                {"role": "system", "content": settings.SYSTEM_PROMPT},
                {"role": "user", "content": formatted_prompt}
            ],
            # Enforce structured JSON output format matching Pydantic schema
            response_format={"type": "json_object"} 
        )
        return response.choices[0].message.content