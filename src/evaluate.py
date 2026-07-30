"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Carrega um ou mais prompts YAML do diretório prompts/
4. Executa prompts contra o dataset
5. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)
- Hugging Face (Qwen/Qwen2.5-7B-Instruct)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, trace, tracing_context

from config import settings
from src.metrics import (
    evaluate_clarity,
    evaluate_f1_score,
    evaluate_precision,
)
from src.utils import (
    check_env_vars,
    format_score,
    get_generation_budget,
    load_yaml,
    print_section_header,
    raise_for_fatal_llm_error,
    set_local_generation_limit,
    trim_simple_generated_answer,
)
from src.utils import (
    get_llm as get_configured_llm,
)


def get_llm():
    return get_configured_llm(temperature=0)


PROMPTS_DIR = settings.BASE_PATH / 'prompts'
DEFAULT_PROMPT = 'bug_to_user_story_v2'


def discover_local_prompts(
    prompts_dir: Path = PROMPTS_DIR,
) -> dict[str, Path]:
    """Descobre as chaves de prompt declaradas nos arquivos YAML locais."""
    discovered: dict[str, Path] = {}
    for prompt_path in sorted(prompts_dir.glob('*.yml')):
        document = load_yaml(str(prompt_path))
        if not isinstance(document, dict):
            continue
        for prompt_name, prompt_data in document.items():
            if not isinstance(prompt_data, dict):
                continue
            if prompt_name in discovered:
                raise ValueError(
                    f"Prompt duplicado '{prompt_name}' em "
                    f'{discovered[prompt_name]} e {prompt_path}',
                )
            discovered[prompt_name] = prompt_path
    return discovered


def load_local_prompt(
    prompt_name: str,
    prompt_paths: dict[str, Path],
) -> ChatPromptTemplate:
    """Carrega a chave selecionada e cria o template usado pela avaliação."""
    prompt_path = prompt_paths.get(prompt_name)
    if prompt_path is None:
        available = ', '.join(sorted(prompt_paths)) or 'nenhum'
        raise ValueError(
            f"Prompt local '{prompt_name}' não encontrado. "
            f'Disponíveis: {available}',
        )

    document = load_yaml(str(prompt_path))
    prompt_data = document.get(prompt_name) if document else None
    if not isinstance(prompt_data, dict):
        raise ValueError(
            f"Prompt '{prompt_name}' inválido em {prompt_path}",
        )

    system_prompt = str(prompt_data.get('system_prompt', '')).strip()
    user_prompt = str(prompt_data.get('user_prompt', '')).strip()
    if not system_prompt or not user_prompt:
        raise ValueError(
            f"Prompt '{prompt_name}' deve conter system_prompt e user_prompt",
        )

    return ChatPromptTemplate.from_messages(
        [('system', system_prompt), ('user', user_prompt)],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta a seleção de prompts informada na linha de comando."""
    parser = argparse.ArgumentParser(
        description='Avalia prompts YAML locais com o dataset do LangSmith.',
    )
    parser.add_argument(
        '-p',
        '--prompt',
        action='append',
        dest='prompts',
        metavar='NOME',
        help=(
            'Chave do prompt local. Repita a opção para comparar versões. '
            f'Padrão: {DEFAULT_PROMPT}.'
        ),
    )
    parser.add_argument(
        '--list-prompts',
        action='store_true',
        help='Lista os prompts encontrados em prompts/ e encerra.',
    )
    return parser.parse_args(argv)


def load_dataset_from_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f'❌ Arquivo não encontrado: {jsonl_path}')
        print(
            '\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.',
        )
        return []
    except json.JSONDecodeError as e:
        print(f'❌ Erro ao parsear JSONL: {e}')
        return []
    except Exception as e:
        print(f'❌ Erro ao carregar dataset: {e}')
        return []


def create_evaluation_dataset(
    client: Client,
    dataset_name: str,
    jsonl_path: str,
) -> str:
    print(f'Criando dataset de avaliação: {dataset_name}...')

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print('❌ Nenhum exemplo carregado do arquivo .jsonl')
        return dataset_name

    print(f'   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}')

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        dataset = client.create_dataset(dataset_name=dataset_name)

        for example in examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs=example['inputs'],
                outputs=example['outputs'],
            )

        print(f'   ✓ Dataset criado com {len(examples)} exemplos')
        return dataset_name

    except Exception as e:
        print(f'   ⚠️  Erro ao criar dataset: {e}')
        return dataset_name


def pull_prompt_from_langsmith(
    prompt_name: str,
    client: Client,
) -> ChatPromptTemplate:
    try:
        print(f'   Puxando prompt do LangSmith Hub: {prompt_name}')
        prompt = client.pull_prompt(
            prompt_name,
            dangerously_pull_public_prompt=True,
        )
        print('   ✓ Prompt carregado com sucesso')
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if 'not found' in error_msg or '404' in error_msg:
            print('⚠️  O prompt não foi encontrado no LangSmith Hub.\n')
            print('AÇÕES NECESSÁRIAS:')
            print('1. Verifique se você já fez push do prompt otimizado:')
            print('   python -m src.push_prompts')
            print()
            print('2. Confirme se o prompt foi publicado com sucesso em:')
            print('   https://smith.langchain.com/prompts')
            print()
            print(
                f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'",
            )
            print()
            print('4. Se você alterou o prompt no YAML, refaça o push:')
            print('   python -m src.push_prompts')
        else:
            print(f'Erro técnico: {e}\n')
            print('Verifique:')
            print('- LANGSMITH_API_KEY está configurada corretamente no .env')
            print('- Você tem acesso ao workspace do LangSmith')
            print('- Sua conexão com a internet está funcionando')

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any,
) -> dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, 'inputs') else {}
        outputs = example.outputs if hasattr(example, 'outputs') else {}

        chain = prompt_template | llm
        set_local_generation_limit(llm, get_generation_budget(inputs))

        response = chain.invoke(inputs)
        answer = trim_simple_generated_answer(response.content, inputs)

        reference = (
            outputs.get('reference', '') if isinstance(outputs, dict) else ''
        )

        if isinstance(inputs, dict):
            question = inputs.get(
                'question',
                inputs.get('bug_report', inputs.get('pr_title', 'N/A')),
            )
        else:
            question = 'N/A'

        return {'answer': answer, 'reference': reference, 'question': question}

    except Exception as e:
        raise_for_fatal_llm_error(e)

        print(f'      ⚠️  Erro ao avaliar exemplo: {e}')
        import traceback

        print(f'      Traceback: {traceback.format_exc()}')
        return {'answer': '', 'reference': '', 'question': ''}


def evaluate_example_with_metrics(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any,
    evaluator_llm: Any,
    client: Client,
    prompt_name: str,
    example_index: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
] | None:
    """Executa geração e métricas dentro de um único trace por exemplo."""
    inputs = example.inputs if hasattr(example, 'inputs') else {}
    project_name = (
        settings.LANGSMITH_PROJECT
        or 'prompt-optimization-challenge-resolved'
    )
    tags = [
        'bug-to-user-story',
        'evaluation',
        f'example-{example_index}',
    ]
    metadata = {
        'prompt_name': prompt_name,
        'llm_provider': settings.LLM_PROVIDER,
        'llm_model': settings.LLM_MODEL,
        'eval_model': settings.EVAL_MODEL,
        'example_index': example_index,
    }

    with tracing_context(
        enabled=settings.LANGSMITH_TRACING,
        project_name=project_name,
        tags=tags,
        metadata=metadata,
        client=client,
    ):
        with trace(
            'evaluate-prompt-example',
            run_type='chain',
            inputs=inputs if isinstance(inputs, dict) else {'input': inputs},
            project_name=project_name,
            tags=tags,
            metadata=metadata,
            client=client,
            reference_example_id=getattr(example, 'id', None),
        ) as root_trace:
            result = evaluate_prompt_on_example(
                prompt_template,
                example,
                llm,
            )
            if not result['answer']:
                root_trace.end(outputs={'status': 'generation-error'})
                return None

            metric_inputs = {
                'question': result['question'],
                'answer': result['answer'],
                'reference': result['reference'],
            }
            with trace(
                'evaluate-f1-score',
                run_type='chain',
                inputs=metric_inputs,
                parent=root_trace,
                client=client,
            ) as metric_trace:
                f1 = evaluate_f1_score(
                    **metric_inputs,
                    llm=evaluator_llm,
                )
                metric_trace.end(outputs=f1)

            with trace(
                'evaluate-clarity',
                run_type='chain',
                inputs=metric_inputs,
                parent=root_trace,
                client=client,
            ) as metric_trace:
                clarity = evaluate_clarity(
                    **metric_inputs,
                    llm=evaluator_llm,
                )
                metric_trace.end(outputs=clarity)

            with trace(
                'evaluate-precision',
                run_type='chain',
                inputs=metric_inputs,
                parent=root_trace,
                client=client,
            ) as metric_trace:
                precision = evaluate_precision(
                    **metric_inputs,
                    llm=evaluator_llm,
                )
                metric_trace.end(outputs=precision)
            root_trace.end(
                outputs={
                    'answer': result['answer'],
                    'f1_score': f1['score'],
                    'clarity': clarity['score'],
                    'precision': precision['score'],
                },
            )
            return result, f1, clarity, precision


def evaluate_prompt(
    prompt_name: str,
    prompt_template: ChatPromptTemplate,
    dataset_name: str,
    client: Client,
) -> dict[str, float]:
    print(f'\n🔍 Avaliando: {prompt_name}')

    try:
        examples = list(client.list_examples(dataset_name=dataset_name))
        print(f'   Dataset: {len(examples)} exemplos')

        llm = get_llm()
        evaluator_llm = (
            llm
            if settings.EVAL_MODEL == settings.LLM_MODEL
            else get_configured_llm(
                model=settings.EVAL_MODEL,
                temperature=0,
            )
        )

        f1_scores = []
        clarity_scores = []
        precision_scores = []
        evaluation_details = []

        print('   Avaliando exemplos...')

        for i, example in enumerate(examples, 1):
            evaluation = evaluate_example_with_metrics(
                prompt_template,
                example,
                llm,
                evaluator_llm,
                client,
                prompt_name,
                i,
            )
            if evaluation is not None:
                result, f1, clarity, precision = evaluation
                f1_scores.append(f1['score'])
                clarity_scores.append(clarity['score'])
                precision_scores.append(precision['score'])

                print(
                    f"      [{i}/{len(examples)}] F1:{f1['score']:.2f} Clarity:{clarity['score']:.2f} Precision:{precision['score']:.2f}",
                )
                metric_results = {
                    'F1': f1,
                    'Clarity': clarity,
                    'Precision': precision,
                }
                evaluation_details.append(
                    {
                        'example': i,
                        'question': result['question'],
                        'answer': result['answer'],
                        'reference': result['reference'],
                        'metrics': {
                            name.lower(): metric_result
                            for name, metric_result in metric_results.items()
                        },
                    },
                )
                for metric_name, metric_result in metric_results.items():
                    if metric_result['score'] >= 0.9:
                        continue
                    reasoning = ' '.join(
                        metric_result.get('reasoning', '').split(),
                    )
                    if reasoning:
                        print(
                            f'         ↳ {metric_name}: {reasoning}',
                        )

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_clarity = (
            sum(clarity_scores) / len(clarity_scores)
            if clarity_scores
            else 0.0
        )
        avg_precision = (
            sum(precision_scores) / len(precision_scores)
            if precision_scores
            else 0.0
        )

        avg_helpfulness = (avg_clarity + avg_precision) / 2
        avg_correctness = (avg_f1 + avg_precision) / 2

        reports_dir = settings.BASE_PATH / 'reports'
        reports_dir.mkdir(exist_ok=True)
        report_name = prompt_name.replace('/', '__') + '.json'
        report_path = reports_dir / report_name
        report_path.write_text(
            json.dumps(
                evaluation_details,
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        print(f'   Relatório detalhado: {report_path}')

        return {
            'helpfulness': round(avg_helpfulness, 4),
            'correctness': round(avg_correctness, 4),
            'f1_score': round(avg_f1, 4),
            'clarity': round(avg_clarity, 4),
            'precision': round(avg_precision, 4),
        }

    except Exception as e:
        print(f'   ❌ Erro na avaliação: {e}')
        return {
            'helpfulness': 0.0,
            'correctness': 0.0,
            'f1_score': 0.0,
            'clarity': 0.0,
            'precision': 0.0,
        }


def display_results(prompt_name: str, scores: dict[str, float]) -> bool:
    print('\n' + '=' * 50)
    print(f'Prompt: {prompt_name}')
    print('=' * 50)

    print('\nMétricas Derivadas:')
    print(
        f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=0.9)}",
    )
    print(
        f"  - Correctness: {format_score(scores['correctness'], threshold=0.9)}",
    )

    print('\nMétricas Base:')
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=0.9)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=0.9)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=0.9)}")

    average_score = sum(scores.values()) / len(scores)

    print('\n' + '-' * 50)
    print(f'📊 MÉDIA GERAL: {average_score:.4f}')
    print('-' * 50)

    all_above_threshold = all(score >= 0.9 for score in scores.values())
    passed = all_above_threshold and average_score >= 0.9

    if passed:
        print('\n✅ STATUS: APROVADO - Todas as métricas >= 0.9')
    else:
        print('\n❌ STATUS: REPROVADO')
        failed_metrics = [
            name for name, score in scores.items() if score < 0.9
        ]
        if failed_metrics:
            print(f"⚠️  Métricas abaixo de 0.9: {', '.join(failed_metrics)}")
        print(f'⚠️  Média atual: {average_score:.4f} | Necessário: 0.9000')

    return passed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt_paths = discover_local_prompts()
    if args.list_prompts:
        print('Prompts disponíveis:')
        for prompt_name, prompt_path in prompt_paths.items():
            print(f'  - {prompt_name}: {prompt_path}')
        return 0

    selected_prompts = args.prompts or [DEFAULT_PROMPT]
    unknown_prompts = [
        prompt_name
        for prompt_name in selected_prompts
        if prompt_name not in prompt_paths
    ]
    if unknown_prompts:
        available = ', '.join(sorted(prompt_paths)) or 'nenhum'
        print(
            f"❌ Prompt(s) não encontrado(s): {', '.join(unknown_prompts)}",
        )
        print(f'   Disponíveis: {available}')
        return 2

    print_section_header('AVALIAÇÃO DE PROMPTS OTIMIZADOS')

    provider = settings.LLM_PROVIDER
    llm_model = settings.LLM_MODEL
    eval_model = settings.EVAL_MODEL

    print(f'Provider: {provider}')
    print(f'Modelo Principal: {llm_model}')
    print(f'Modelo de Avaliação: {eval_model}\n')

    if provider == 'hugging_face' and settings.HF_MAX_NEW_TOKENS > 1536:
        print(
            '⚠️  HF_MAX_NEW_TOKENS acima de 1536 pode prolongar respostas '
            'repetitivas do modelo local e reduzir Clarity/Precision.\n',
        )

    required_vars = ['LANGSMITH_API_KEY', 'LLM_PROVIDER']
    if provider == 'openai':
        required_vars.append('OPENAI_API_KEY')
    elif provider in ['google', 'gemini']:
        required_vars.append('GOOGLE_API_KEY')
    elif provider == 'hugging_face':
        if settings.HF_EXECUTION_MODE == 'inference':
            required_vars.append('HUGGING_FACE_API_KEY')

    if not check_env_vars(required_vars):
        return 1

    client = Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url=settings.LANGSMITH_ENDPOINT,
    )
    project_name = (
        settings.LANGSMITH_PROJECT or 'prompt-optimization-challenge-resolved'
    )

    jsonl_path = settings.BASE_PATH / 'datasets/bug_to_user_story.jsonl'

    if not Path(jsonl_path).exists():
        print(f'❌ Arquivo de dataset não encontrado: {jsonl_path}')
        print('\nCertifique-se de que o arquivo existe antes de continuar.')
        return 1

    dataset_name = f'{project_name}-eval'
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print('\n' + '=' * 70)
    print('PROMPTS PARA AVALIAR')
    print('=' * 70)
    print('\nEste script utilizará os prompts YAML locais:')
    for prompt_name in selected_prompts:
        print(f'  - {prompt_name}: {prompt_paths[prompt_name]}')
    print()

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in selected_prompts:
        evaluated_count += 1

        try:
            prompt_template = load_local_prompt(prompt_name, prompt_paths)
            scores = evaluate_prompt(
                prompt_name,
                prompt_template,
                dataset_name,
                client,
            )

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append(
                {'prompt': prompt_name, 'scores': scores, 'passed': passed},
            )

        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append(
                {
                    'prompt': prompt_name,
                    'scores': {
                        'helpfulness': 0.0,
                        'correctness': 0.0,
                        'f1_score': 0.0,
                        'clarity': 0.0,
                        'precision': 0.0,
                    },
                    'passed': False,
                },
            )

    print('\n' + '=' * 50)
    print('RESUMO FINAL')
    print('=' * 50 + '\n')

    if evaluated_count == 0:
        print('⚠️  Nenhum prompt foi avaliado')
        return 1

    print(f'Prompts avaliados: {evaluated_count}')
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(
        f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n",
    )

    if all_passed:
        print('✅ Todos os prompts atingiram todas as métricas >= 0.9!')
        print('\n✓ Confira os resultados em:')
        print(f'  https://smith.langchain.com/projects/{project_name}')
        print('\nPróximos passos:')
        print('1. Documente o processo no README.md')
        print('2. Capture screenshots das avaliações')
        print('3. Faça commit e push para o GitHub')
        return 0
    print('⚠️  Alguns prompts não atingiram todas as métricas >= 0.9')
    print('\nPróximos passos:')
    print('1. Refatore os prompts com score baixo')
    print('2. Edite o arquivo YAML local correspondente')
    print('3. Execute: python -m src.evaluate --prompt NOME novamente')
    return 1


if __name__ == '__main__':
    sys.exit(main())
