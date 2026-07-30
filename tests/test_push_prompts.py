"""Testes da publicação de prompts no LangSmith."""

from unittest.mock import Mock

from langsmith.utils import LangSmithError

from src.push_prompts import (
    is_idempotent_push_error,
    normalize_tag,
    push_prompt_to_langsmith,
)


def test_normalize_tag_removes_unsupported_characters() -> None:
    """Pontos, espaços e acentos devem gerar uma tag válida."""
    assert normalize_tag('v2.1') == 'v2-1'
    assert normalize_tag('Técnica avançada / CoT') == 'Tecnica-avancada-CoT'


def test_push_uses_normalized_commit_tags() -> None:
    """Nenhuma commit tag enviada pode conter caracteres inválidos."""
    client = Mock()
    client.push_prompt.return_value = (
        'https://smith.langchain.com/prompts/test'
    )
    prompt_data = {
        'system_prompt': 'Você é um Product Manager. Exemplo',
        'user_prompt': '{bug_report}',
        'description': 'Prompt de teste',
        'version': 'v2.1',
        'techniques_applied': ['Few-shot Learning'],
        'tags': ['product/management'],
    }

    result = push_prompt_to_langsmith(
        'owner/prompt',
        prompt_data,
        client=client,
    )

    assert result is True
    call = client.push_prompt.call_args
    assert call.kwargs['commit_tags'] == [
        'v2-1',
        'product-management',
        'few-shot-learning',
    ]


def test_push_succeeds_when_prompt_has_not_changed() -> None:
    """Repetir o push do mesmo conteúdo deve ser uma operação idempotente."""
    client = Mock()
    client.push_prompt.side_effect = LangSmithError(
        '409 Conflict: Nothing to commit: prompt has not changed since '
        'latest commit',
    )
    prompt_data = {
        'system_prompt': 'Você é um Product Manager. Exemplo',
        'user_prompt': '{bug_report}',
        'description': 'Prompt de teste',
        'version': 'v2.1',
        'techniques_applied': [],
        'tags': [],
    }

    result = push_prompt_to_langsmith(
        'owner/prompt',
        prompt_data,
        client=client,
    )

    assert result is True


def test_existing_commit_tag_is_idempotent() -> None:
    """Tag já associada ao commit significa que o estado remoto foi aplicado."""
    error = LangSmithError(
        '409 Conflict: Tag bug-analysis already exists on commit',
    )

    assert is_idempotent_push_error(error) is True


def test_push_succeeds_when_commit_tag_already_exists() -> None:
    """Conflito ao reaplicar tag não deve interromper make iterate."""
    client = Mock()
    client.push_prompt.side_effect = LangSmithError(
        '409 Conflict: Tag bug-analysis already exists on commit',
    )
    prompt_data = {
        'system_prompt': 'Você é um Product Manager. Exemplo',
        'user_prompt': '{bug_report}',
        'description': 'Prompt de teste',
        'version': 'v2.2',
        'techniques_applied': [],
        'tags': ['bug-analysis'],
    }

    result = push_prompt_to_langsmith(
        'owner/prompt',
        prompt_data,
        client=client,
    )

    assert result is True


def test_push_still_fails_for_other_langsmith_errors() -> None:
    """Somente o conflito idempotente deve ser convertido em sucesso."""
    client = Mock()
    client.push_prompt.side_effect = LangSmithError('Unauthorized')
    prompt_data = {
        'system_prompt': 'Você é um Product Manager. Exemplo',
        'user_prompt': '{bug_report}',
        'description': 'Prompt de teste',
        'version': 'v2.1',
        'techniques_applied': [],
        'tags': [],
    }

    result = push_prompt_to_langsmith(
        'owner/prompt',
        prompt_data,
        client=client,
    )

    assert result is False
