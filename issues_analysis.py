from typing import List

ISSUES_PROMPT_HEADER = """
Проанализируй тексты анализов собраний региона {region} за последние {count} собраний и выдели до 4 основных отраслей, которые обсуждались.

Инструкция:
1. Изучи все анализы собраний.
2. Выдели до 4 наиболее часто упоминаемых и значимых отраслей. Если их меньше, укажи только те, что действительно обсуждались.
3. Для каждой отрасли укажи контекст обсуждения.
4. Если для отрасли есть проблема или задача, которая встречается абсолютно на всех собраниях, укажи её. Если такой нет, оставь только отрасль и ничего не выдумывай.

Формат ответа строго:
Основные обсуждаемые отрасли:

1. [Название отрасли]
   • [Краткое описание проблем и вопросов]
   • [Общая для всех собраний проблема или задача, если есть]

Если отрасли не выделяются:
"На основе анализа материалов собраний не удалось выделить конкретные отрасли для обсуждения."

Анализы собраний:
{separator}
"""

GET_RECENT_ANALYSES = """
    SELECT va.analysis_text, va.created_at
    FROM video_analyses va
    WHERE va.group_code = $1
    ORDER BY va.created_at DESC
    LIMIT $2
"""


def build_issues_prompt(analyses: List, region: str, count: int) -> str:
    prompt = ISSUES_PROMPT_HEADER.format(region=region, count=count, separator="-" * 50)
    for i, item in enumerate(analyses, 1):
        prompt += f"\nСобрание {i} ({item['created_at']:%d.%m.%Y}):\n{item['analysis_text']}\n{'-' * 30}\n"
    return prompt


async def analyze_recent_meetings(db, gigachat, group_code: str, region: str, count: int) -> str:
    """Сводный анализ отраслей по последним count собраниям региона."""
    total = await db.fetchval("SELECT COUNT(*) FROM video_analyses WHERE group_code = $1", group_code)
    if count > total:
        raise ValueError(f"Доступно только {total} собраний")

    analyses = await db.fetch(GET_RECENT_ANALYSES, group_code, count)
    return await gigachat.send_prompt(build_issues_prompt(analyses, region, count))