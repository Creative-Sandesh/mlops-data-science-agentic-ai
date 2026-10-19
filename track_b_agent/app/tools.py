def get_current_weather(location: str) -> str:
    """Mock weather tool function."""
    return f"The current weather in {location} is 22°C and sunny."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]