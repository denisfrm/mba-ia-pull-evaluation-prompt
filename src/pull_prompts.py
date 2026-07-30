"""Baixa o prompt inicial do LangSmith e o salva no formato YAML do projeto."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langsmith import Client
from langsmith.utils import LangSmithError

from config import settings
from src.utils import print_section_header, save_yaml

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts.chat import BaseMessagePromptTemplate

    PromptMessage = BaseMessagePromptTemplate | BaseMessage

DEFAULT_OUTPUT = settings.BASE_PATH / settings.LANGSMITH_LOCAL_PROMPT


def _message_role(message: PromptMessage) -> str | None:
    """Converte uma mensagem do LangChain para a chave usada no YAML."""
    name = type(message).__name__.lower()
    if 'system' in name:
        return 'system_prompt'
    if 'human' in name or 'user' in name:
        return 'user_prompt'
    return None


def _message_template(message: PromptMessage) -> str:
    prompt = getattr(message, 'prompt', None)
    template = getattr(prompt, 'template', None)
    if isinstance(template, str):
        return template

    content = getattr(message, 'content', None)
    if isinstance(content, str):
        return content

    message_type = type(message).__name__
    error = f'Tipo de mensagem não suportado: {message_type}'
    raise ValueError(error)


def prompt_to_yaml(prompt_template: ChatPromptTemplate) -> dict[str, Any]:
    """Extrai mensagens system/user de um ChatPromptTemplate."""
    prompt_data: dict[str, Any] = {
        'description': 'Prompt inicial obtido do LangSmith Prompt Hub',
        'version': 'v1',
        'tags': ['bug-analysis', 'user-story', 'product-management'],
    }

    for message in getattr(prompt_template, 'messages', []):
        role = _message_role(message)
        if role and role not in prompt_data:
            prompt_data[role] = _message_template(message)

    if not prompt_data.get('system_prompt') or not prompt_data.get(
        'user_prompt',
    ):
        error = 'O prompt remoto deve conter mensagens system e user'
        raise ValueError(error)

    return {'bug_to_user_story_v1': prompt_data}


def pull_prompts_from_langsmith(
    client: Client,
    prompt_name: str = settings.LANGSMITH_PULL_PROMPT,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """Faz pull de um prompt público e persiste sua representação local."""
    prompt_template = client.pull_prompt(
        prompt_name,
        dangerously_pull_public_prompt=True,
    )
    if not save_yaml(prompt_to_yaml(prompt_template), str(output_path)):
        error = f'Não foi possível salvar o prompt em {output_path}'
        raise OSError(error)
    return output_path


def main() -> int:
    print_section_header(
        'Iniciando pull de prompts do LangSmith Prompt Hub',
    )

    if not settings.LANGSMITH_API_KEY.strip():
        print('Erro: LANGSMITH_API_KEY não configurada no arquivo .env.')
        return 1

    prompt_name = settings.LANGSMITH_PULL_PROMPT
    output = Path(settings.LANGSMITH_LOCAL_PROMPT)
    if not output.is_absolute():
        output = settings.BASE_PATH / output

    try:
        saved_path = pull_prompts_from_langsmith(
            Client(
                api_key=settings.LANGSMITH_API_KEY,
                api_url=settings.LANGSMITH_ENDPOINT,
            ),
            prompt_name,
            output,
        )
    except (LangSmithError, OSError, ValueError) as exc:
        print(f'Erro ao puxar prompt: {exc}')
        return 1

    print(f"Prompt '{prompt_name}' salvo em: {saved_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
