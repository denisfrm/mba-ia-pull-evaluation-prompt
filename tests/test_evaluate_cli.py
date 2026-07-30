"""Testes da seleção de prompts locais na avaliação."""

from pathlib import Path

import pytest

from src.evaluate import (
    discover_local_prompts,
    load_local_prompt,
    parse_args,
)


def write_prompt(path: Path, prompt_name: str) -> None:
    """Cria um prompt YAML mínimo para os testes."""
    path.write_text(
        f'''{prompt_name}:
  system_prompt: "Converta o bug em uma User Story."
  user_prompt: "Relato: {{bug_report}}"
''',
        encoding='utf-8',
    )


def test_discover_and_load_local_prompt(tmp_path: Path) -> None:
    """A chave YAML deve ser descoberta e convertida em template."""
    prompt_path = tmp_path / 'prompt.yml'
    write_prompt(prompt_path, 'bug_to_user_story_test')

    prompts = discover_local_prompts(tmp_path)
    template = load_local_prompt('bug_to_user_story_test', prompts)

    assert prompts == {'bug_to_user_story_test': prompt_path}
    assert template.input_variables == ['bug_report']


def test_duplicate_prompt_names_are_rejected(tmp_path: Path) -> None:
    """Duas versões não podem declarar a mesma chave de prompt."""
    write_prompt(tmp_path / 'first.yml', 'duplicate')
    write_prompt(tmp_path / 'second.yml', 'duplicate')

    with pytest.raises(ValueError, match='Prompt duplicado'):
        discover_local_prompts(tmp_path)


def test_parse_multiple_prompts() -> None:
    """A opção --prompt pode ser repetida para comparar versões."""
    args = parse_args(
        [
            '--prompt',
            'bug_to_user_story_v1',
            '-p',
            'bug_to_user_story_v2',
        ],
    )

    assert args.prompts == [
        'bug_to_user_story_v1',
        'bug_to_user_story_v2',
    ]


def test_parse_list_prompts() -> None:
    """A listagem deve funcionar sem iniciar uma avaliação."""
    args = parse_args(['--list-prompts'])

    assert args.list_prompts is True
