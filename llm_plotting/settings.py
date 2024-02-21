from pydantic_settings import BaseSettings
from pydantic import BaseModel


class Settings(BaseSettings):
    openai_api_key: str
    e2b_api_key: str

    class Config:
        env_file = ".env"
        extra = "allow"


class AgentSettings(BaseModel):
    max_iterations: int = 5
    code_generation_llm_temperature: float = 0
    image_validation_llm_temperature: float = 0
