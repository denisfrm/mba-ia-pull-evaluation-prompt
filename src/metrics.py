"""
Módulo de métricas customizadas para avaliação de prompts.

Este módulo implementa métricas gerais e específicas para Bug to User Story:

MÉTRICAS GERAIS (3):
1. F1-Score: Balanceamento entre Precision e Recall
2. Clarity: Clareza e estrutura da resposta
3. Precision: Informações corretas e relevantes

MÉTRICAS ESPECÍFICAS PARA BUG TO USER STORY (4):
4. Tone Score: Tom profissional e empático
5. Acceptance Criteria Score: Qualidade dos critérios de aceitação
6. User Story Format Score: Formato correto (Como... Eu quero... Para que...)
7. Completeness Score: Completude e contexto técnico

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage

from config import settings
from src.utils import (
    get_eval_llm,
    raise_for_fatal_llm_error,
    set_local_generation_limit,
)

RATING_SCORES = {
    'excellent': 1.0,
    'acceptable': 0.9,
    'insufficient': 0.7,
    'incorrect': 0.0,
}


def rating_to_score(rating: Any) -> float:
    """Converte a classificação ordinal do juiz em score reproduzível."""
    normalized = str(rating).strip().lower()
    return RATING_SCORES.get(normalized, 0.0)


def get_evaluator_llm():
    """
    Retorna o LLM configurado para avaliação.
    Suporta OpenAI e Google Gemini baseado no .env
    """
    return get_eval_llm(temperature=0)


def extract_json_from_response(response_text: str) -> dict[str, Any]:
    """
    Extrai JSON de uma resposta de LLM que pode conter texto adicional.
    """
    try:
        # Tentar parsear diretamente
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Tentar encontrar JSON no meio do texto
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start != -1 and end > start:
            try:
                json_str = response_text[start:end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Se não conseguir extrair, retornar valores default
        print(
            f'⚠️  Não foi possível extrair JSON da resposta: {response_text[:200]}...',
        )
        return {'score': 0.0, 'reasoning': 'Erro ao processar resposta'}


def evaluate_f1_score(
    question: str,
    answer: str,
    reference: str,
    llm: Any | None = None,
) -> dict[str, Any]:
    """
    Calcula F1-Score usando LLM-as-Judge.

    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        question: Pergunta feita pelo usuário
        answer: Resposta gerada pelo prompt
        reference: Resposta esperada (ground truth)

    Returns:
        Dict com score e reasoning:
        {
            "score": 0.95,
            "precision": 0.9,
            "recall": 0.99,
            "reasoning": "Explicação do LLM..."
        }

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em medir a qualidade de respostas geradas por IA.

Sua tarefa é calcular PRECISION e RECALL para determinar o F1-Score.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA ESPERADA (Ground Truth):
{reference}

RESPOSTA GERADA PELO MODELO:
{answer}

INSTRUÇÕES:

1. PRECISION:
   - Avalie equivalência semântica, não coincidência literal de palavras.
   - Considere correta uma informação sustentada pela pergunta OU pela
     referência.
   - Penalize somente afirmações contraditórias, irrelevantes ou ausentes tanto
     da pergunta quanto da referência.

2. RECALL:
   - Quantas informações da resposta esperada estão PRESENTES na resposta gerada?
   - Aceite paráfrases e soluções tecnicamente equivalentes.
   - Penalizar informações importantes que foram omitidas

3. RACIOCÍNIO:
   - Antes de penalizar, cite a afirmação exata e confira novamente pergunta,
     referência e resposta.
   - Não afirme que um detalhe está ausente da pergunta sem verificar o texto.

CLASSIFICAÇÃO OBRIGATÓRIA PARA PRECISION E RECALL:
- "excellent": cobertura completa, sem problema relevante.
- "acceptable": conteúdo central correto; apenas diferenças menores que não
  exigem reescrever a resposta.
- "insufficient": omissão ou adição importante que exige edição substancial.
- "incorrect": resposta centralmente errada ou fora do tema.

Se seu raciocínio usar "diferenças menores", "pequenas divergências" ou
"captura a maioria", a classificação deve ser no mínimo "acceptable".

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "precision_rating": "<excellent|acceptable|insufficient|incorrect>",
  "recall_rating": "<excellent|acceptable|insufficient|incorrect>",
  "reasoning": "<sua explicação em até 100 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        evaluator_llm = llm or get_evaluator_llm()
        set_local_generation_limit(evaluator_llm, 512)
        response = evaluator_llm.invoke(
            [HumanMessage(content=evaluator_prompt)],
        )
        result = extract_json_from_response(response.content)

        precision_rating = result.get('precision_rating', '')
        recall_rating = result.get('recall_rating', '')
        precision = rating_to_score(precision_rating)
        recall = rating_to_score(recall_rating)

        # Calcular F1-Score
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        return {
            'score': round(f1_score, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'precision_rating': precision_rating,
            'recall_rating': recall_rating,
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar F1-Score: {e}')
        return {
            'score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'reasoning': f'Erro na avaliação: {e!s}',
        }


def evaluate_clarity(
    question: str,
    answer: str,
    reference: str,
    llm: Any | None = None,
) -> dict[str, Any]:
    """
    Avalia a clareza e estrutura da resposta usando LLM-as-Judge.

    Critérios:
    - Organização e estrutura clara
    - Linguagem simples e direta
    - Ausência de ambiguidade
    - Fácil de entender

    Args:
        question: Pergunta feita pelo usuário
        answer: Resposta gerada pelo prompt
        reference: Resposta esperada (ground truth)

    Returns:
        Dict com score e reasoning:
        {
            "score": 0.92,
            "reasoning": "Explicação do LLM..."
        }

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em medir a CLAREZA de respostas geradas por IA.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA GERADA PELO MODELO:
{answer}

RESPOSTA ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie a CLAREZA da resposta gerada com base nos critérios:

1. ORGANIZAÇÃO (0.0 a 1.0):
   - A resposta tem estrutura lógica e bem organizada?
   - Informações estão em ordem sensata?

2. LINGUAGEM (0.0 a 1.0):
   - Usa linguagem simples e direta?
   - Evita jargões desnecessários?
   - Fácil de entender?

3. AUSÊNCIA DE AMBIGUIDADE (0.0 a 1.0):
   - A resposta é clara e sem ambiguidades?
   - Não deixa dúvidas sobre o que está sendo comunicado?

4. CONCISÃO (0.0 a 1.0):
   - É concisa sem ser curta demais?
   - Não tem informações redundantes?

Calcule a MÉDIA dos 4 critérios para obter o score final.

CALIBRAÇÃO:
- 0.90 a 1.00: clara e utilizável; pequenas redundâncias não comprometem.
- 0.80 a 0.89: há problema perceptível que exige edição.
- Abaixo de 0.80: repetição, ambiguidade ou desorganização substancial.
- Não reduza abaixo de 0.90 apenas por preferência de estilo. Para reduzir,
  cite um trecho específico que prejudique a compreensão.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação detalhada da avaliação em até 100 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        evaluator_llm = llm or get_evaluator_llm()
        set_local_generation_limit(evaluator_llm, 512)
        response = evaluator_llm.invoke(
            [HumanMessage(content=evaluator_prompt)],
        )
        result = extract_json_from_response(response.content)

        score = float(result.get('score', 0.0))

        return {
            'score': round(score, 4),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar Clarity: {e}')
        return {'score': 0.0, 'reasoning': f'Erro na avaliação: {e!s}'}


def evaluate_precision(
    question: str,
    answer: str,
    reference: str,
    llm: Any | None = None,
) -> dict[str, Any]:
    """
    Avalia a precisão da resposta usando LLM-as-Judge.

    Critérios:
    - Ausência de informações inventadas (alucinações)
    - Resposta focada na pergunta
    - Informações corretas e verificáveis

    Args:
        question: Pergunta feita pelo usuário
        answer: Resposta gerada pelo prompt
        reference: Resposta esperada (ground truth)

    Returns:
        Dict com score e reasoning:
        {
            "score": 0.98,
            "reasoning": "Explicação do LLM..."
        }

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em detectar PRECISÃO e ALUCINAÇÕES em respostas de IA.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA GERADA PELO MODELO:
{answer}

RESPOSTA ESPERADA (Ground Truth):
{reference}

INSTRUÇÕES:

Avalie a PRECISÃO da resposta gerada:

1. AUSÊNCIA DE ALUCINAÇÕES:
   - A resposta contém informações INVENTADAS ou não verificáveis?
   - A referência é ground truth autoritativa: qualquer informação presente na
     pergunta OU na referência é permitida e não é alucinação.
   - Consequências necessárias e paráfrases equivalentes também são permitidas.

2. FOCO NA PERGUNTA:
   - A resposta responde EXATAMENTE o que foi perguntado?
   - Não divaga ou adiciona informações não solicitadas?

3. CORREÇÃO FACTUAL:
   - As informações estão CORRETAS quando comparadas com a referência?
   - Não há erros ou imprecisões?

VERIFICAÇÃO OBRIGATÓRIA:
- Antes de chamar algo de alucinação, copie mentalmente a afirmação e procure
  suporte tanto na pergunta quanto na referência.
- Nunca diga que um número, endpoint, plataforma ou requisito está ausente sem
  revisar os dois textos.
- Não penalize uma seção técnica apenas por ela ser técnica; penalize somente
  conteúdo incorreto, irrelevante ou sem suporte.

CLASSIFICAÇÃO OBRIGATÓRIA:
- "excellent": fiel, focada e sem problema relevante.
- "acceptable": correta no conteúdo central; somente diferenças menores.
- "insufficient": contém imprecisão ou divagação importante.
- "incorrect": centralmente errada, inventada ou fora do tema.

Se o raciocínio disser "geralmente precisa", "bem focada", "diferença menor"
ou "pequena divergência", classifique como "acceptable", não "insufficient".

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "rating": "<excellent|acceptable|insufficient|incorrect>",
  "reasoning": "<explicação detalhada em até 100 palavras, cite exemplos>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        evaluator_llm = llm or get_evaluator_llm()
        set_local_generation_limit(evaluator_llm, 512)
        response = evaluator_llm.invoke(
            [HumanMessage(content=evaluator_prompt)],
        )
        result = extract_json_from_response(response.content)

        rating = result.get('rating', '')
        score = rating_to_score(rating)

        return {
            'score': round(score, 4),
            'rating': rating,
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar Precision: {e}')
        return {'score': 0.0, 'reasoning': f'Erro na avaliação: {e!s}'}


def evaluate_tone_score(
    bug_report: str,
    user_story: str,
    reference: str,
) -> dict[str, Any]:
    """
    Avalia o tom da user story (profissional e empático).

    Critérios específicos para Bug to User Story:
    - Tom profissional mas não excessivamente técnico
    - Empatia com o usuário afetado pelo bug
    - Foco em valor de negócio, não apenas correção técnica
    - Linguagem positiva (o que o usuário QUER fazer, não só o que não funciona)

    Args:
        bug_report: Descrição do bug original
        user_story: User story gerada pelo prompt
        reference: User story esperada (ground truth)

    Returns:
        Dict com score e reasoning

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em User Stories ágeis.

BUG REPORT ORIGINAL:
{bug_report}

USER STORY GERADA:
{user_story}

USER STORY ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie o TOM da user story gerada com base nos critérios:

1. PROFISSIONALISMO (0.0 a 1.0):
   - Usa linguagem profissional e apropriada para documentação?
   - Evita jargões excessivos ou linguagem muito informal?
   - Mantém padrão de qualidade de documentação ágil?

2. EMPATIA COM USUÁRIO (0.0 a 1.0):
   - Demonstra compreensão do impacto do bug no usuário?
   - Foca na necessidade/frustração do usuário?
   - Usa linguagem centrada no usuário ("Como um... eu quero...")?

3. FOCO EM VALOR (0.0 a 1.0):
   - Articula claramente o valor de negócio da solução?
   - Vai além de "consertar o bug" e explica o benefício?
   - Usa a estrutura "para que eu possa..." com valor real?

4. LINGUAGEM POSITIVA (0.0 a 1.0):
   - Foca no que o usuário QUER fazer (não só no que está quebrado)?
   - Tom construtivo e orientado a solução?
   - Evita linguagem negativa ou culpabilizante?

Calcule a MÉDIA dos 4 critérios para obter o score final.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação detalhada em até 150 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)

        score = float(result.get('score', 0.0))

        return {
            'score': round(score, 4),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar Tone Score: {e}')
        return {'score': 0.0, 'reasoning': f'Erro na avaliação: {e!s}'}


def evaluate_acceptance_criteria_score(
    bug_report: str,
    user_story: str,
    reference: str,
) -> dict[str, Any]:
    """
    Avalia a qualidade dos critérios de aceitação.

    Critérios específicos:
    - Usa formato Given-When-Then ou similar estruturado
    - Critérios são específicos e testáveis
    - Quantidade adequada (3-7 critérios idealmente)
    - Cobertura completa do bug e solução
    - Incluem cenários de edge case quando relevante

    Args:
        bug_report: Descrição do bug original
        user_story: User story gerada pelo prompt
        reference: User story esperada (ground truth)

    Returns:
        Dict com score e reasoning

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em Critérios de Aceitação de User Stories.

BUG REPORT ORIGINAL:
{bug_report}

USER STORY GERADA:
{user_story}

USER STORY ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie os CRITÉRIOS DE ACEITAÇÃO da user story gerada:

1. FORMATO ESTRUTURADO (0.0 a 1.0):
   - Usa formato Given-When-Then ou estrutura similar?
   - Cada critério é claramente separado e identificável?
   - Formatação facilita leitura e entendimento?

2. ESPECIFICIDADE E TESTABILIDADE (0.0 a 1.0):
   - Critérios são específicos e não vagos?
   - É possível criar testes automatizados a partir deles?
   - Evita termos ambíguos como "deve funcionar bem"?
   - Critérios mensuráveis e verificáveis?

3. QUANTIDADE ADEQUADA (0.0 a 1.0):
   - Tem quantidade apropriada de critérios (nem muito, nem pouco)?
   - Ideal: 3-7 critérios para bugs simples/médios
   - Bugs complexos podem ter mais critérios organizados

4. COBERTURA COMPLETA (0.0 a 1.0):
   - Cobre todos os aspectos do bug?
   - Inclui cenários de sucesso e erro?
   - Considera edge cases quando relevante?
   - Aborda validações e requisitos técnicos do bug?

Calcule a MÉDIA dos 4 critérios para obter o score final.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação detalhada com exemplos específicos, até 150 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)

        score = float(result.get('score', 0.0))

        return {
            'score': round(score, 4),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar Acceptance Criteria Score: {e}')
        return {'score': 0.0, 'reasoning': f'Erro na avaliação: {e!s}'}


def evaluate_user_story_format_score(
    bug_report: str,
    user_story: str,
    reference: str,
) -> dict[str, Any]:
    """
    Avalia se a user story segue o formato padrão correto.

    Formato esperado:
    - "Como um [tipo de usuário]"
    - "Eu quero [ação/funcionalidade]"
    - "Para que [benefício/valor]"
    - Critérios de Aceitação claramente separados

    Args:
        bug_report: Descrição do bug original
        user_story: User story gerada pelo prompt
        reference: User story esperada (ground truth)

    Returns:
        Dict com score e reasoning

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em formato de User Stories ágeis.

BUG REPORT ORIGINAL:
{bug_report}

USER STORY GERADA:
{user_story}

USER STORY ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie o FORMATO da user story gerada:

1. TEMPLATE PADRÃO (0.0 a 1.0):
   - Segue o formato "Como um [usuário], eu quero [ação], para que [benefício]"?
   - Todas as três partes estão presentes e corretas?
   - Ordem e estrutura seguem as melhores práticas?

2. IDENTIFICAÇÃO DE PERSONA (0.0 a 1.0):
   - "Como um..." identifica claramente o tipo de usuário?
   - Persona é específica e relevante para o bug?
   - Evita genéricos como "Como um usuário" sem contexto?

3. AÇÃO CLARA (0.0 a 1.0):
   - "Eu quero..." descreve claramente a ação/funcionalidade desejada?
   - Ação é específica e relacionada ao bug?
   - Evita descrições vagas ou muito técnicas?

4. BENEFÍCIO ARTICULADO (0.0 a 1.0):
   - "Para que..." explica claramente o valor/benefício?
   - Benefício é real e significativo (não trivial)?
   - Conecta a ação ao valor de negócio?

5. SEPARAÇÃO DE SEÇÕES (0.0 a 1.0):
   - User story principal está claramente separada dos critérios?
   - Critérios de aceitação têm seção própria?
   - Estrutura facilita leitura e navegação?

Calcule a MÉDIA dos 5 critérios para obter o score final.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação detalhada com exemplos, até 150 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)

        score = float(result.get('score', 0.0))

        return {
            'score': round(score, 4),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar User Story Format Score: {e}')
        return {'score': 0.0, 'reasoning': f'Erro na avaliação: {e!s}'}


def evaluate_completeness_score(
    bug_report: str,
    user_story: str,
    reference: str,
) -> dict[str, Any]:
    """
    Avalia a completude da user story em relação ao bug.

    Critérios específicos baseados na complexidade do bug:
    - Bugs simples: cobre o problema básico
    - Bugs médios: inclui contexto técnico relevante
    - Bugs complexos: aborda múltiplos aspectos, impacto, tasks técnicas

    Args:
        bug_report: Descrição do bug original
        user_story: User story gerada pelo prompt
        reference: User story esperada (ground truth)

    Returns:
        Dict com score e reasoning

    """
    evaluator_prompt = f"""
Você é um avaliador especializado em completude de User Stories derivadas de bugs.

BUG REPORT ORIGINAL:
{bug_report}

USER STORY GERADA:
{user_story}

USER STORY ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie a COMPLETUDE da user story em relação ao bug:

1. COBERTURA DO PROBLEMA (0.0 a 1.0):
   - A user story aborda TODOS os aspectos do bug reportado?
   - Nenhum detalhe importante foi omitido?
   - Se bug menciona múltiplos problemas, todos são cobertos?

2. CONTEXTO TÉCNICO (0.0 a 1.0):
   - Quando o bug inclui detalhes técnicos (logs, stack traces, endpoints):
     * User story preserva contexto técnico relevante?
     * Informações técnicas são incluídas de forma apropriada?
   - Bugs simples não precisam de muito contexto técnico
   - Bugs complexos DEVEM incluir seção de contexto técnico

3. IMPACTO E SEVERIDADE (0.0 a 1.0):
   - Se o bug menciona impacto (usuários afetados, perda financeira):
     * User story reconhece e documenta o impacto?
   - Severidade é refletida na priorização implícita?
   - Bugs críticos devem ter tratamento mais detalhado

4. TASKS TÉCNICAS (0.0 a 1.0):
   - Para bugs complexos com múltiplos componentes:
     * User story sugere tasks técnicas ou breakdown?
   - Para bugs simples/médios:
     * Tasks não são necessárias (não penalizar ausência)
   - Avalie se o nível de detalhe é apropriado à complexidade

5. INFORMAÇÕES ADICIONAIS RELEVANTES (0.0 a 1.0):
   - Se bug menciona: steps to reproduce, ambiente, logs
     * User story preserva ou referencia essas informações?
   - Contexto de negócio importante é mantido?
   - Sugestões de solução são apropriadas?

Calcule a MÉDIA dos 5 critérios para obter o score final.

IMPORTANTE:
- Bugs SIMPLES podem ter score alto mesmo sem muitos detalhes técnicos
- Bugs COMPLEXOS DEVEM ter seções adicionais (contexto técnico, tasks, impacto)
- Compare com a referência para calibrar expectativa de completude

Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação detalhada sobre o que foi bem coberto e o que faltou, até 200 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""

    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)

        score = float(result.get('score', 0.0))

        return {
            'score': round(score, 4),
            'reasoning': result.get('reasoning', ''),
        }

    except Exception as e:
        raise_for_fatal_llm_error(e)
        print(f'❌ Erro ao avaliar Completeness Score: {e}')
        return {'score': 0.0, 'reasoning': f'Erro na avaliação: {e!s}'}


# Exemplo de uso e testes
if __name__ == '__main__':
    # Mostrar provider configurado
    provider = settings.LLM_PROVIDER
    eval_model = settings.EVAL_MODEL

    print('=' * 70)
    print('TESTANDO MÉTRICAS CUSTOMIZADAS')
    print('=' * 70)
    print(f'\n📊 Provider: {provider}')
    print(f'🤖 Modelo de Avaliação: {eval_model}\n')

    print('=' * 70)
    print('PARTE 1: MÉTRICAS GERAIS')
    print('=' * 70)

    # Teste das métricas gerais
    test_question = 'Qual o horário de funcionamento da loja?'
    test_answer = 'A loja funciona de segunda a sexta das 9h às 18h.'
    test_reference = 'Horário de funcionamento: Segunda a Sexta 9:00-18:00, Sábado 9:00-14:00'

    print('\n1. F1-Score:')
    f1_result = evaluate_f1_score(test_question, test_answer, test_reference)
    print(f"   Score: {f1_result['score']:.2f}")
    print(f"   Precision: {f1_result['precision']:.2f}")
    print(f"   Recall: {f1_result['recall']:.2f}")
    print(f"   Reasoning: {f1_result['reasoning']}\n")

    print('2. Clarity:')
    clarity_result = evaluate_clarity(
        test_question,
        test_answer,
        test_reference,
    )
    print(f"   Score: {clarity_result['score']:.2f}")
    print(f"   Reasoning: {clarity_result['reasoning']}\n")

    print('3. Precision:')
    precision_result = evaluate_precision(
        test_question,
        test_answer,
        test_reference,
    )
    print(f"   Score: {precision_result['score']:.2f}")
    print(f"   Reasoning: {precision_result['reasoning']}\n")

    print('=' * 70)
    print('PARTE 2: MÉTRICAS ESPECÍFICAS PARA BUG TO USER STORY')
    print('=' * 70)

    # Teste das métricas específicas de Bug to User Story
    test_bug = (
        'Botão de adicionar ao carrinho não funciona no produto ID 1234.'
    )
    test_user_story = """Como um cliente navegando na loja, eu quero adicionar produtos ao meu carrinho de compras, para que eu possa continuar comprando e finalizar minha compra depois.

Critérios de Aceitação:
- Dado que estou visualizando um produto
- Quando clico no botão "Adicionar ao Carrinho"
- Então o produto deve ser adicionado ao carrinho
- E devo ver uma confirmação visual
- E o contador do carrinho deve ser atualizado"""

    test_reference_story = test_user_story  # Usando o mesmo para teste

    print('\n4. Tone Score (Tom profissional e empático):')
    tone_result = evaluate_tone_score(
        test_bug,
        test_user_story,
        test_reference_story,
    )
    print(f"   Score: {tone_result['score']:.2f}")
    print(f"   Reasoning: {tone_result['reasoning']}\n")

    print('5. Acceptance Criteria Score (Qualidade dos critérios):')
    criteria_result = evaluate_acceptance_criteria_score(
        test_bug,
        test_user_story,
        test_reference_story,
    )
    print(f"   Score: {criteria_result['score']:.2f}")
    print(f"   Reasoning: {criteria_result['reasoning']}\n")

    print('6. User Story Format Score (Formato correto):')
    format_result = evaluate_user_story_format_score(
        test_bug,
        test_user_story,
        test_reference_story,
    )
    print(f"   Score: {format_result['score']:.2f}")
    print(f"   Reasoning: {format_result['reasoning']}\n")

    print('7. Completeness Score (Completude e contexto):')
    completeness_result = evaluate_completeness_score(
        test_bug,
        test_user_story,
        test_reference_story,
    )
    print(f"   Score: {completeness_result['score']:.2f}")
    print(f"   Reasoning: {completeness_result['reasoning']}\n")

    print('=' * 70)
    print('✅ TODOS OS TESTES CONCLUÍDOS!')
    print('=' * 70)
