import time
from typing import Optional

from crewai import Agent, Task, Crew, LLM
from app.logger import logger
from app.settings import settings

# Инициализация LLM (общая для всех агентов)
llm = LLM(
    model=settings.OLLAMA_MODEL,
    base_url=settings.ollama_url,
    temperature=0.0,
    timeout=60.0,          # Таймаут на один HTTP-запрос к модели
    max_tokens=300,
    api_key=settings.API_KEY
)


def _create_generator_agent():
    return Agent(
        role='API-документатор',
        goal='Генерировать документацию в строгом формате.',
        backstory=(
            'Ты специалист по API. Генерируй документацию ТОЛЬКО в формате:\n'
            '### МЕТОД /путь\n'
            '**Описание**: ... \n'
            '**Параметры**: ... \n'
            '**Ответ**:\n'
            '```json\n{...}\n```'
        ),
        llm=llm,
        verbose=False
    )


def _create_validator_agent():
    return Agent(
        role='Валидатор документации',
        goal='Проверять соответствие документа строгому формату.',
        backstory=(
            'Ты строгий QA-инженер. Ты должен проверить, что документ содержит:\n'
            '1. Строку вида "### МЕТОД /путь",\n'
            '2. Блок "**Описание**:",\n'
            '3. Блок "**Параметры" или "**Параметры пути**:",\n'
            '4. Блок "**Ответ**:" с JSON-примером в тройных кавычках.\n'
            'Если всё есть — ответь ТОЛЬКО "valid". Иначе — ТОЛЬКО "invalid".'
        ),
        llm=llm,
        verbose=False
    )


def generate_and_validate_documentation(query: str) -> Optional[str]:
    """
    Генерирует документацию и проверяет её валидатором.
    Возвращает контент, только если он валиден.
    """
    logger.info(f'[DOC-GEN] Старт обработки запроса: "{query}"')

    # === ЭТАП 1: Генерация ===
    generator = _create_generator_agent()
    gen_task = Task(
        description=f'Создай документацию для: {query}',
        expected_output='Документ в строгом формате. Ничего больше.',
        agent=generator
    )
    gen_crew = Crew(
        agents=[generator],
        tasks=[gen_task],
        max_iterations=5,   # Защита от зацикливания даже для одного агента
        verbose=False
    )

    raw_content = None
    latency_gen = None

    try:
        start = time.perf_counter()
        raw_content = str(gen_crew.kickoff()).strip()
        latency_gen = time.perf_counter() - start
        logger.debug(
            f'[DOC-GEN] Генерация завершена. Latency={latency_gen:.3f}s. '
            f'Длина результата={len(raw_content)} символов.'
        )
    except Exception as e:
        logger.exception(f'[DOC-GEN] Критическая ошибка на этапе генерации: {e}')
        return None

    if not raw_content:
        logger.error(f'[DOC-GEN] Пустой результат после генерации для запроса: "{query}".')
        return None

    # === ЭТАП 2: Валидация ===
    validator = _create_validator_agent()
    val_task = Task(
        description=f'Проверь документ:\n\n{raw_content}',
        expected_output='Только "valid" или "invalid", без пояснений.',
        agent=validator
    )
    val_crew = Crew(
        agents=[validator],
        tasks=[val_task],
        max_iterations=3,   # Валидация должна быть быстрой
        verbose=False
    )

    validation_result = None
    latency_val = None

    try:
        start = time.perf_counter()
        validation_result = str(val_crew.kickoff()).strip().lower()
        latency_val = time.perf_counter() - start
        logger.debug(
            f'[DOC-GEN] Валидация завершена. Результат="{validation_result}", Latency={latency_val:.3f}s.'
        )
    except Exception as e:
        logger.exception(f'[DOC-GEN] Критическая ошибка на этапе валидации: {e}')
        return None

    total_latency = (latency_gen or 0) + (latency_val or 0)
    logger.info(
        f'[DOC-GEN] Полный цикл завершён для запроса: "{query}". '
        f'Gen={latency_gen:.3f}s | Val={latency_val:.3f}s | Total={total_latency:.3f}s'
    )

    # Строгое сравнение: только точное совпадение
    if validation_result == 'valid':
        logger.info(f'[DOC-GEN] Документация успешно валидирована для запроса: "{query}".')
        return raw_content
    else:
        logger.warning(
            f'[DOC-GEN] Валидация не пройдена: результат="{validation_result}" для запроса: "{query}".'
        )
        return None