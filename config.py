from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BASE_PATH: Path = Path(__file__).resolve().parent
    model_config = SettingsConfigDict(
        env_file=BASE_PATH / '.env',
        env_file_encoding='utf-8',
    )

    # LangSmith Configuration
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = 'https://api.smith.langchain.com'
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str = ''
    LANGSMITH_PULL_PROMPT: str = 'leonanluppi/bug_to_user_story_v1'
    LANGSMITH_LOCAL_PROMPT: str = 'prompts/bug_to_user_story_v1.yml'
    USERNAME_LANGSMITH_HUB: str

    # OpenAI Configuration
    OPENAI_API_KEY: str | None = None

    # Google Gemini Configuration
    GOOGLE_API_KEY: str | None = None

    # LLM Configuration
    LLM_PROVIDER: Literal['openai', 'google', 'hugging_face'] = 'google'
    LLM_MODEL: str = 'gemini-2.5-flash'
    EVAL_MODEL: str = 'gemini-2.5-flash'

    # Hugging Face config
    HUGGING_FACE_API_KEY: str | None = None
    HF_EXECUTION_MODE: Literal['inference', 'local'] = 'inference'
    HF_INFERENCE_PROVIDER: str = 'auto'
    HF_MAX_NEW_TOKENS: int = 1024
    HF_EMBEDDING_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'

settings = Settings()
