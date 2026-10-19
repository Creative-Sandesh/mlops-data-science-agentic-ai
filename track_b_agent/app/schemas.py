from pydantic import BaseModel, Field

class WeatherToolInput(BaseModel):
    location:str = Field(description=" The city and state, eg kathmandu, Nepal" )
    
class AssistantStructuredResponse(BaseModel):
    answer: str = Field(description="Direct, accurate response to the query")
    sources_used: list[str] = Field(default=[], description="List of retrieved context snippets used")
    confidence_score: float = Field(description="Confidence level between 0.0 and 1.0")
