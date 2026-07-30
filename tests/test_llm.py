"""Testes da configuração dos providers de LLM."""

from unittest.mock import Mock, patch

import pytest
from transformers import GenerationConfig

from config import settings
from src.utils import (
    LLMProviderError,
    get_eval_llm,
    get_generation_budget,
    get_llm,
    raise_for_fatal_llm_error,
    set_local_generation_limit,
    trim_simple_generated_answer,
)


def test_hugging_face_returns_native_chat_model(monkeypatch):
    """Hugging Face deve manter o contrato de chat usado pelas chains."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    monkeypatch.setattr(settings, 'HUGGING_FACE_API_KEY', 'hf_test')
    monkeypatch.setattr(settings, 'LLM_MODEL', 'owner/main-model')
    monkeypatch.setattr(settings, 'HF_EXECUTION_MODE', 'inference')
    monkeypatch.setattr(settings, 'HF_INFERENCE_PROVIDER', 'auto')
    monkeypatch.setattr(settings, 'HF_MAX_NEW_TOKENS', 512)
    endpoint = Mock()
    chat_model = Mock()

    with (
        patch(
            'langchain_huggingface.HuggingFaceEndpoint',
            return_value=endpoint,
        ) as endpoint_class,
        patch(
            'langchain_huggingface.ChatHuggingFace',
            return_value=chat_model,
        ) as chat_class,
    ):
        result = get_llm(temperature=0)

    assert result is chat_model
    endpoint_class.assert_called_once_with(
        repo_id='owner/main-model',
        task='text-generation',
        provider='auto',
        huggingfacehub_api_token='hf_test',
        temperature=0,
        max_new_tokens=512,
        do_sample=False,
    )
    chat_class.assert_called_once_with(
        llm=endpoint,
        temperature=0,
        max_tokens=512,
    )


def test_hugging_face_passes_sampling_configuration(monkeypatch):
    """Temperatura positiva deve habilitar amostragem no endpoint nativo."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    monkeypatch.setattr(settings, 'HUGGING_FACE_API_KEY', 'hf_test')
    monkeypatch.setattr(settings, 'HF_EXECUTION_MODE', 'inference')

    with (
        patch(
            'langchain_huggingface.HuggingFaceEndpoint',
        ) as endpoint_class,
        patch('langchain_huggingface.ChatHuggingFace') as chat_class,
    ):
        get_llm(model='owner/model', temperature=0.3)

    assert endpoint_class.call_args.kwargs['temperature'] == 0.3
    assert endpoint_class.call_args.kwargs['do_sample'] is True
    assert chat_class.call_args.kwargs['temperature'] == 0.3


def test_hugging_face_uses_evaluation_model(monkeypatch):
    """O avaliador deve poder usar um modelo diferente do gerador."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    monkeypatch.setattr(settings, 'HUGGING_FACE_API_KEY', 'hf_test')
    monkeypatch.setattr(settings, 'HF_EXECUTION_MODE', 'inference')
    monkeypatch.setattr(settings, 'EVAL_MODEL', 'owner/evaluator')

    with (
        patch(
            'langchain_huggingface.HuggingFaceEndpoint',
        ) as endpoint_class,
        patch('langchain_huggingface.ChatHuggingFace'),
    ):
        get_eval_llm()

    assert endpoint_class.call_args.kwargs['repo_id'] == 'owner/evaluator'


def test_hugging_face_requires_api_key(monkeypatch):
    """Token ausente deve produzir erro antes de criar o endpoint."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    monkeypatch.setattr(settings, 'HUGGING_FACE_API_KEY', None)
    monkeypatch.setattr(settings, 'HF_EXECUTION_MODE', 'inference')

    with pytest.raises(ValueError, match='HUGGING_FACE_API_KEY'):
        get_llm()


def test_hugging_face_can_run_locally_without_api_key(monkeypatch):
    """Modo local deve usar pipeline nativo sem acessar o Inference Router."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    monkeypatch.setattr(settings, 'HUGGING_FACE_API_KEY', None)
    monkeypatch.setattr(settings, 'HF_EXECUTION_MODE', 'local')
    monkeypatch.setattr(settings, 'LLM_MODEL', 'owner/local-model')
    monkeypatch.setattr(settings, 'HF_MAX_NEW_TOKENS', 256)
    pipeline = Mock()
    pipeline.pipeline.model.generation_config = GenerationConfig(
        eos_token_id=42,
        pad_token_id=42,
    )
    chat_model = Mock()

    with (
        patch(
            'langchain_huggingface.HuggingFacePipeline.from_model_id',
            return_value=pipeline,
        ) as from_model_id,
        patch(
            'langchain_huggingface.ChatHuggingFace',
            return_value=chat_model,
        ) as chat_class,
    ):
        result = get_llm()

    assert result is chat_model
    from_model_id.assert_called_once()
    call_kwargs = from_model_id.call_args.kwargs
    assert call_kwargs['model_id'] == 'owner/local-model'
    assert call_kwargs['task'] == 'text-generation'
    generation_config = pipeline.pipeline.generation_config
    assert generation_config.max_length is None
    assert generation_config.max_new_tokens == 256
    assert generation_config.do_sample is False
    assert generation_config.temperature is None
    assert generation_config.top_p is None
    assert generation_config.top_k is None
    assert generation_config.eos_token_id == 42
    assert generation_config.pad_token_id == 42
    assert (
        call_kwargs['pipeline_kwargs']['clean_up_tokenization_spaces'] is False
    )
    assert call_kwargs['pipeline_kwargs']['return_full_text'] is False
    assert 'generation_config' not in call_kwargs['pipeline_kwargs']
    assert pipeline.pipeline.model.generation_config is generation_config
    chat_class.assert_called_once_with(llm=pipeline)


def test_hugging_face_credit_error_is_fatal(monkeypatch):
    """Saldo esgotado deve interromper novas chamadas ao provider."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    error = RuntimeError(
        "402 Payment Required: You have depleted your monthly included credits."
    )

    with pytest.raises(LLMProviderError, match='créditos.*esgotados'):
        raise_for_fatal_llm_error(error)


def test_non_hugging_face_error_is_not_reclassified(monkeypatch):
    """Erros de outros providers não devem ser alterados pelo helper."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'google')

    assert (
        raise_for_fatal_llm_error(RuntimeError('402 Payment Required')) is None
    )


def test_local_generation_limit_updates_loaded_pipeline(monkeypatch):
    """O orçamento deve mudar sem carregar outra cópia do modelo."""
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'hugging_face')
    monkeypatch.setattr(settings, 'HF_EXECUTION_MODE', 'local')
    llm = Mock()
    llm.llm.pipeline.generation_config.max_new_tokens = 1024
    llm.llm.pipeline.model.generation_config.max_new_tokens = 1024

    set_local_generation_limit(llm, 256)

    assert llm.llm.pipeline.generation_config.max_new_tokens == 256
    assert llm.llm.pipeline.model.generation_config.max_new_tokens == 256


def test_simple_generation_is_trimmed_after_fifth_criterion():
    """Rótulos e cenários extras não podem contaminar respostas simples."""
    inputs = {'bug_report': 'Botão não funciona.'}
    answer = """Como um usuário, eu quero usar o botão, para concluir a ação.

Critérios de Aceitação:
- Dado que vejo o botão
- Quando clico no botão
- Então a ação deve ocorrer
- E devo ver uma confirmação
- E o estado deve ser atualizado

=== COMPLEXIDADE ===
SIMPLES
- Cenário inventado"""

    result = trim_simple_generated_answer(answer, inputs)

    assert get_generation_budget(inputs) == 256
    assert result.endswith('- E o estado deve ser atualizado')
    assert 'COMPLEXIDADE' not in result
    assert 'Cenário inventado' not in result
