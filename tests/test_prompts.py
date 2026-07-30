"""Testes automatizados para validação de prompts."""

# ruff: noqa: I001, S101

from pathlib import Path
from typing import Any

import pytest
import yaml

from config import settings
from src.utils import validate_prompt_structure

PROMPT_PATH = settings.BASE_PATH / 'prompts' / 'bug_to_user_story_v2.yml'


PromptData = dict[str, Any]
MINIMUM_EXAMPLES = 2
MINIMUM_TECHNIQUES = 2


def load_prompts(file_path: Path) -> PromptData:
    """Carrega prompts do arquivo YAML."""
    with file_path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def prompt_data() -> PromptData:
    """Retorna os dados da versão otimizada."""
    document = load_prompts(PROMPT_PATH)
    assert 'bug_to_user_story_v2' in document
    return document['bug_to_user_story_v2']


class TestPrompts:
    def test_prompt_has_system_prompt(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert prompt_data.get('system_prompt', '').strip()

    def test_prompt_has_role_definition(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Verifica se o prompt define uma persona especializada."""
        system_prompt = prompt_data['system_prompt'].lower()
        assert 'você é um product manager' in system_prompt

    def test_prompt_mentions_format(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt_data['system_prompt'].lower()
        assert 'markdown' in system_prompt
        assert 'como um [tipo de usuário]' in system_prompt

    def test_prompt_has_few_shot_examples(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Verifica exemplos de entrada/saída para Few-shot."""
        system_prompt = prompt_data['system_prompt'].lower()
        assert system_prompt.count('entrada:') >= MINIMUM_EXAMPLES
        assert system_prompt.count('saída:') >= MINIMUM_EXAMPLES
        assert any(
            technique.lower() == 'few-shot learning'
            for technique in prompt_data['techniques_applied']
        )

    def test_prompt_no_todos(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        assert '[TODO]' not in yaml.safe_dump(
            prompt_data,
            allow_unicode=True,
        )

    def test_minimum_techniques(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Verifica se pelo menos duas técnicas foram listadas."""
        techniques = prompt_data.get('techniques_applied', [])
        assert len(techniques) >= MINIMUM_TECHNIQUES
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, errors

    def test_prompt_adapts_output_to_bug_complexity(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Garante cobertura distinta para bugs simples, médios e complexos."""
        system_prompt = prompt_data['system_prompt']
        assert 'SIMPLES' in system_prompt
        assert 'MÉDIO' in system_prompt
        assert 'COMPLEXO' in system_prompt
        assert '=== CRITÉRIOS TÉCNICOS ===' in system_prompt
        assert '=== CONTEXTO DO BUG ===' in system_prompt

    def test_prompt_requires_atomic_fact_coverage(
        self,
        prompt_data: PromptData,
    ) -> None:
        """Evita perda dos detalhes que derruba recall e F1."""
        system_prompt = prompt_data['system_prompt'].lower()
        assert 'cada fato extraído' in system_prompt
        assert 'nenhum problema' in system_prompt
        assert 'não invente' in system_prompt


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
