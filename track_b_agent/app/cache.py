import redis
import json
import hashlib
from app.config import settings

class CacheService:
    def __init__(self):
        # Connects to Redis container in docker-compose
        self.redis_client = redis.Redis(
            host=getattr(settings, "REDIS_HOST", "redis"),
            port=6379,
            db=0,
            decode_responses=True
        )

    def _generate_key(self, prompt: str) -> str:
        return f"cache:{hashlib.sha256(prompt.strip().lower().encode()).hexdigest()}"

    def get_response(self, prompt: str):
        try:
            cached_data = self.redis_client.get(self._generate_key(prompt))
            return json.loads(cached_data) if cached_data else None
        except Exception:
            return None  # Fail gracefully if cache is down

    def set_response(self, prompt: str, response_data: dict, ttl: int = 3600):
        try:
            key = self._generate_key(prompt)
            self.redis_client.setex(key, ttl, json.dumps(response_data))
        except Exception:
            pass