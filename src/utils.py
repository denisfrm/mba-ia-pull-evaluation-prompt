"""
Funções auxiliares para o projeto de otimização de prompts.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from config import settings

MAX_TECHNIQUES = 2


class LLMProviderError(RuntimeError):
    """Erro fatal do provider que não será resolvido repetindo a chamada."""


def raise_for_fatal_llm_error(error: Exception) -> None:
    """Converte erros fatais do provider em mensagens acionáveis."""
    if settings.LLM_PROVIDER.lower() != 'hugging_face':
        return

    error_message = str(error)
    credits_exhausted = (
        '402 Payment Required' in error_message
        or 'depleted your monthly included credits' in error_message
    )
    if credits_exhausted:
        raise LLMProviderError(
            'Os créditos do Hugging Face Inference Providers estão esgotados. '
            'A avaliação de 15 exemplos pode realizar até 60 requisições '
            '(15 gerações e 45 julgamentos). Adicione créditos à conta, '
            'aguarde a renovação da franquia ou configure '
            'HF_EXECUTION_MODE=local; '
            'trocar apenas o modelo/provider não remove o bloqueio 402.',
        ) from error

    unsupported_model = (
        'model_not_supported' in error_message
        or "doesn't support task 'conversational'" in error_message
    )
    if unsupported_model:
        raise LLMProviderError(
            f"O modelo '{settings.LLM_MODEL}' não está disponível no provider "
            f"'{settings.HF_INFERENCE_PROVIDER}'. Escolha uma combinação "
            'disponível em https://huggingface.co/inference/models.',
        ) from error


def set_local_generation_limit(llm: Any, max_new_tokens: int) -> None:
    """Ajusta o limite do pipeline local já carregado, sem recriar o modelo."""
    if (
        settings.LLM_PROVIDER.lower() != 'hugging_face'
        or settings.HF_EXECUTION_MODE != 'local'
    ):
        return

    native_pipeline = getattr(getattr(llm, 'llm', None), 'pipeline', None)
    if native_pipeline is None:
        return

    native_pipeline.generation_config.max_new_tokens = max_new_tokens
    native_pipeline.model.generation_config.max_new_tokens = max_new_tokens


def get_generation_budget(inputs: Any) -> int:
    """Define um teto proporcional à complexidade aparente do bug."""
    if not isinstance(inputs, dict):
        return 768

    bug_report = str(
        inputs.get('bug_report', inputs.get('question', '')),
    ).strip()
    if '\n' not in bug_report and len(bug_report) <= 300:
        return 256
    if len(bug_report) >= 1200 or bug_report.count('\n') >= 15:
        return 1024
    return 768


def trim_simple_generated_answer(answer: str, inputs: Any) -> str:
    """Remove conteúdo que o modelo acrescentar após o quinto critério."""
    if get_generation_budget(inputs) != 256:
        return answer.strip()

    lines = answer.splitlines()
    bullet_indexes = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith('- ')
    ]
    if len(bullet_indexes) < 5:
        return answer.strip()

    return '\n'.join(lines[: bullet_indexes[4] + 1]).strip()


def load_yaml(file_path: str) -> dict[str, Any] | None:
    """
    Carrega arquivo YAML.

    Args:
        file_path: Caminho do arquivo YAML

    Returns:
        Dicionário com conteúdo do YAML ou None se erro

    """
    try:
        with open(file_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f'❌ Arquivo não encontrado: {file_path}')
        return None
    except yaml.YAMLError as e:
        print(f'❌ Erro ao parsear YAML: {e}')
        return None
    except Exception as e:
        print(f'❌ Erro ao carregar arquivo: {e}')
        return None


def save_yaml(data: dict[str, Any], file_path: str) -> bool:
    """
    Salva dados em arquivo YAML.

    Args:
        data: Dados para salvar
        file_path: Caminho do arquivo de saída

    Returns:
        True se sucesso, False caso contrário

    """
    try:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)

        return True
    except Exception as e:
        print(f'❌ Erro ao salvar arquivo: {e}')
        return False


def check_env_vars(required_vars: list) -> bool:
    """
    Verifica se variáveis de ambiente obrigatórias estão configuradas.

    Args:
        required_vars: Lista de variáveis obrigatórias

    Returns:
        True se todas configuradas, False caso contrário

    """
    missing_vars = []

    for var in required_vars:
        value = getattr(settings, var, None)
        if value is None or not str(value).strip():
            missing_vars.append(var)

    if missing_vars:
        print('❌ Variáveis de ambiente faltando:')
        for var in missing_vars:
            print(f'   - {var}')
        print('\nConfigure-as no arquivo .env antes de continuar.')
        return False

    return True


def format_score(score: float, threshold: float = 0.9) -> str:
    """
    Formata score com indicador visual de aprovação.

    Args:
        score: Score entre 0.0 e 1.0
        threshold: Limite mínimo para aprovação

    Returns:
        String formatada com score e símbolo

    """
    symbol = '✓' if score >= threshold else '✗'
    return f'{score:.2f} {symbol}'


def print_section_header(title: str, char: str = '=', width: int = 50):
    """
    Imprime cabeçalho de seção formatado.

    Args:
        title: Título da seção
        char: Caractere para a linha
        width: Largura da linha

    """
    print('\n' + char * width)
    print(title)
    print(char * width + '\n')


def validate_prompt_structure(
    prompt_data: dict[str, Any],
) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt.

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros

    """
    errors = []

    required_fields = ['description', 'system_prompt', 'version']
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f'Campo obrigatório faltando: {field}')

    system_prompt = prompt_data.get('system_prompt', '').strip()
    if not system_prompt:
        errors.append('system_prompt está vazio')

    if 'TODO' in system_prompt:
        errors.append('system_prompt ainda contém TODOs')

    techniques = prompt_data.get('techniques_applied', [])
    if len(techniques) < MAX_TECHNIQUES:
        errors.append(
            f'Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}',
        )

    return (len(errors) == 0, errors)


def extract_json_from_response(response_text: str) -> dict[str, Any] | None:
    """
    Extrai JSON de uma resposta de LLM que pode conter texto adicional.

    Args:
        response_text: Texto da resposta do LLM

    Returns:
        Dicionário extraído ou None se não encontrar JSON válido

    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start != -1 and end > start:
            try:
                json_str = response_text[start:end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return None


def get_llm(model: str | None = None, temperature: float = 0.0):
    """
    Retorna uma instância de LLM configurada baseada no provider.

    Args:
        model: Nome do modelo (opcional, usa LLM_MODEL do .env por padrão)
        temperature: Temperatura para geração (padrão: 0.0 para determinístico)

    Returns:
        Instância de ChatOpenAI ou ChatGoogleGenerativeAI

    Raises:
        ValueError: Se provider não for suportado ou API key não configurada

    """
    provider = settings.LLM_PROVIDER.lower()
    model_name = model or settings.LLM_MODEL

    if provider == 'openai':
        from langchain_openai import ChatOpenAI

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                'OPENAI_API_KEY não configurada no .env\n'
                'Obtenha uma chave em: https://platform.openai.com/api-keys',
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError(
                'GOOGLE_API_KEY não configurada no .env\n'
                'Obtenha uma chave em: https://aistudio.google.com/app/apikey',
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
    if provider == 'hugging_face':
        from langchain_huggingface import (
            ChatHuggingFace,
            HuggingFaceEndpoint,
            HuggingFacePipeline,
        )

        if settings.HF_EXECUTION_MODE == 'local':
            try:
                pipeline = HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    task='text-generation',
                    pipeline_kwargs={
                        'clean_up_tokenization_spaces': False,
                        'return_full_text': False,
                    },
                )
                generation_config = deepcopy(
                    pipeline.pipeline.model.generation_config,
                )
                generation_config.max_length = None
                generation_config.max_new_tokens = settings.HF_MAX_NEW_TOKENS
                generation_config.do_sample = temperature > 0
                generation_config.temperature = (
                    temperature if temperature > 0 else None
                )
                generation_config.top_p = None
                generation_config.top_k = None
                pipeline.pipeline.generation_config = generation_config
                pipeline.pipeline.model.generation_config = generation_config
            except ImportError as error:
                raise ValueError(
                    'O modo local requer transformers, torch e accelerate. '
                    'Instale com: uv sync --extra hugging-face-local',
                ) from error

            return ChatHuggingFace(
                llm=pipeline,
            )

        api_key = settings.HUGGING_FACE_API_KEY
        if not api_key:
            raise ValueError(
                'HUGGING_FACE_API_KEY não configurada no .env\n'
                'Obtenha uma chave em: https://huggingface.co/settings/tokens',
            )

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            task='text-generation',
            provider=settings.HF_INFERENCE_PROVIDER,
            huggingfacehub_api_token=api_key,
            temperature=temperature,
            max_new_tokens=settings.HF_MAX_NEW_TOKENS,
            do_sample=temperature > 0,
        )
        return ChatHuggingFace(
            llm=endpoint,
            temperature=temperature,
            max_tokens=settings.HF_MAX_NEW_TOKENS,
        )

    raise ValueError(
        f"Provider '{provider}' não suportado.\n"
        "Use 'openai', 'google' ou 'hugging_face' na variável "
        "LLM_PROVIDER do .env",
    )


def get_eval_llm(temperature: float = 0.0):
    """
    Retorna LLM configurado especificamente para avaliação (usa EVAL_MODEL).

    Args:
        temperature: Temperatura para geração

    Returns:
        Instância de LLM configurada para avaliação

    """
    return get_llm(model=settings.EVAL_MODEL, temperature=temperature)
