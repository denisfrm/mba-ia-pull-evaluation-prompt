"""Valida e publica o prompt otimizado no LangSmith Prompt Hub."""

import re
import sys
import unicodedata
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.utils import LangSmithError

from config import settings
from src.utils import (
    load_yaml,
    print_section_header,
    validate_prompt_structure,
)

PROMPT_PATH = settings.BASE_PATH / 'prompts' / 'bug_to_user_story_v2.yml'
PROMPT_KEY = 'bug_to_user_story_v2'


def normalize_tag(value: Any) -> str:
    """Converte metadados em tags aceitas pela API do LangSmith."""
    ascii_value = (
        unicodedata.normalize('NFKD', str(value))
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '-', ascii_value)
    normalized = normalized.strip('-_')
    if not normalized:
        raise ValueError(
            f'Não foi possível criar uma tag válida para {value!r}',
        )
    return normalized


def is_idempotent_push_error(error: Exception) -> bool:
    """Identifica conflitos que indicam que o estado remoto já foi aplicado."""
    error_message = str(error)
    return 'Nothing to commit' in error_message or (
        'Tag ' in error_message
        and ' already exists on commit' in error_message
    )


def build_prompt_template(prompt_data: dict[str, Any]) -> ChatPromptTemplate:
    """Cria o objeto serializável esperado pelo Prompt Hub."""
    return ChatPromptTemplate.from_messages(
        [
            ('system', prompt_data['system_prompt']),
            ('user', prompt_data['user_prompt']),
        ],
    )


def push_prompt_to_langsmith(
    prompt_name: str,
    prompt_data: dict[str, Any],
    client: Client | None = None,
) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt
        client: Cliente injetável para permitir testes sem acesso à rede.

    Returns:
        True se sucesso, False caso contrário

    """
    langsmith_client = client or Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url=settings.LANGSMITH_ENDPOINT,
    )
    techniques = prompt_data.get('techniques_applied', [])
    tags = list(
        dict.fromkeys(
            [
                *(normalize_tag(tag) for tag in prompt_data.get('tags', [])),
                *(
                    normalize_tag(technique).lower()
                    for technique in techniques
                ),
            ],
        ),
    )
    commit_tags = list(
        dict.fromkeys(
            [
                normalize_tag(prompt_data['version']),
                *tags,
            ],
        ),
    )
    commit_description = (
        'Técnicas aplicadas: ' + ', '.join(techniques) if techniques else None
    )

    try:
        url = langsmith_client.push_prompt(
            prompt_name,
            object=build_prompt_template(prompt_data),
            is_public=True,
            description=prompt_data['description'],
            tags=tags,
            commit_tags=commit_tags,
            commit_description=commit_description,
        )
    except LangSmithError as exc:
        if is_idempotent_push_error(exc):
            print(
                f"Prompt '{prompt_name}' já está atualizado no LangSmith; "
                'nenhuma alteração para publicar.',
            )
            return True
        print(f"Erro ao publicar '{prompt_name}': {exc}")
        return False

    print(f'Prompt publicado com sucesso: {url}')
    return True


def validate_prompt(prompt_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros

    """
    is_valid, errors = validate_prompt_structure(prompt_data)

    user_prompt = prompt_data.get('user_prompt', '').strip()
    if not user_prompt:
        errors.append('user_prompt está vazio')
    elif '{bug_report}' not in user_prompt:
        errors.append('user_prompt deve conter a variável {bug_report}')

    if 'Exemplo' not in prompt_data.get('system_prompt', ''):
        errors.append('system_prompt deve conter exemplos Few-shot')

    return not errors and is_valid, errors


def main() -> int:
    """Carrega, valida e publica a versão v2."""
    print_section_header('PUBLICAÇÃO DO PROMPT OTIMIZADO')

    if not settings.LANGSMITH_API_KEY.strip():
        print('Erro: LANGSMITH_API_KEY não configurada no arquivo .env.')
        return 1
    if not settings.USERNAME_LANGSMITH_HUB.strip():
        print('Erro: USERNAME_LANGSMITH_HUB não configurado no arquivo .env.')
        return 1

    document = load_yaml(str(PROMPT_PATH))
    if not document or PROMPT_KEY not in document:
        print(f"Prompt '{PROMPT_KEY}' não encontrado em {PROMPT_PATH}")
        return 1

    prompt_data = document[PROMPT_KEY]
    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print('Prompt inválido:')
        for error in errors:
            print(f'  - {error}')
        return 1

    username = settings.USERNAME_LANGSMITH_HUB.strip().strip('/')
    prompt_name = f'{username}/{PROMPT_KEY}'
    client = Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url=settings.LANGSMITH_ENDPOINT,
    )
    return (
        0
        if push_prompt_to_langsmith(
            prompt_name,
            prompt_data,
            client,
        )
        else 1
    )


if __name__ == '__main__':
    sys.exit(main())
