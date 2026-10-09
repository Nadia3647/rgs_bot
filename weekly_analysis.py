import logging
import os
import tempfile
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

COMMON_ISSUES_PROMPT = """
Цель: на основе данных выявить от 1 до 3 ключевых проблем, задач или целей, абсолютно общих для КАЖДОГО из упомянутых регионов.

Главное правило: если хотя бы для одного региона нет явного текстового подтверждения проблемы, она НЕ считается общей.

Критерии:
1. Конкретность. Ищи одинаковые конкретные формулировки (например, "ремонт дорог", "поддержка семей мобилизованных"). Обобщающие категории ("развитие инфраструктуры", "социальная политика") запрещены.
2. Текстуальное подтверждение. Основание только явное упоминание идентичной задачи в контексте каждого региона. Не домысливай.

Порядок действий:
1. Выдели конкретные проблемы и задачи по каждому региону.
2. Для каждой потенциально общей проверь, упоминается ли она в КАЖДОМ регионе без исключения.
3. Сформируй ответ в одном из двух форматов.

ВАРИАНТ 1 (общие задачи найдены):
Выявленные общие проблемы/задачи:
1. [Единая для всех формулировка задачи №1]
2. [...]
3. [...]

ВАРИАНТ 2 (не найдены, выводи ТОЛЬКО эту строку):
Результат анализа: Общих для всех регионов проблем и задач выявлено не было.
"""

GET_WEEKLY_ANALYSES = """
    SELECT g.region_name, g.group_code, v.video_link, v.analysis_text, v.created_at
    FROM video_analyses v
    JOIN video_groups g ON v.group_code = g.group_code
    WHERE v.created_at BETWEEN $1 AND $2
      AND g.group_code <> ALL($3::text[])
    ORDER BY g.region_name, v.created_at DESC
"""


def week_bounds(offset_weeks: int = 0) -> Tuple[date, date]:
    """Понедельник и воскресенье недели; offset_weeks=-1 даёт предыдущую неделю."""
    today = datetime.now().date()
    start = today - timedelta(days=today.weekday()) + timedelta(weeks=offset_weeks)
    return start, start + timedelta(days=6)


async def get_cached_analysis(db, week_start: date, week_end: date) -> Optional[str]:
    row = await db.fetchrow(
        "SELECT gigachat_analysis_text FROM weekly_analyses_cache "
        "WHERE week_start_date = $1 AND week_end_date = $2",
        week_start, week_end,
    )
    return row["gigachat_analysis_text"] if row else None


async def save_analysis_to_cache(db, week_start: date, week_end: date, text: str) -> None:
    await db.execute(
        """
        INSERT INTO weekly_analyses_cache (week_start_date, week_end_date, gigachat_analysis_text)
        VALUES ($1, $2, $3)
        ON CONFLICT (week_start_date, week_end_date)
        DO UPDATE SET gigachat_analysis_text = EXCLUDED.gigachat_analysis_text,
                      cache_created_at = NOW()
        """,
        week_start, week_end, text,
    )


def render_analyses_file(analyses: List, week_start: date, week_end: date) -> str:
    lines = [
        f"Анализы собраний за период {week_start:%d.%m.%Y} - {week_end:%d.%m.%Y}",
        f"Количество собраний: {len(analyses)}",
        "=" * 50, "",
    ]
    current_region = None
    for item in analyses:
        if item["region_name"] != current_region:
            current_region = item["region_name"]
            lines += ["", f"### {current_region} ({item['group_code']})", ""]
        lines += [f"Дата: {item['created_at']:%d.%m.%Y %H:%M}",
                  f"Анализ:\n{item['analysis_text']}", "-" * 40]
    return "\n".join(lines)


async def get_common_issues(db, gigachat, excluded_groups: List[str], offset_weeks: int = 0) -> Optional[str]:
    """Возвращает общие проблемы регионов за неделю, используя кэш из БД."""
    week_start, week_end = week_bounds(offset_weeks)

    cached = await get_cached_analysis(db, week_start, week_end)
    if cached:
        return cached

    analyses = await db.fetch(GET_WEEKLY_ANALYSES, week_start, week_end + timedelta(days=1), excluded_groups)
    if not analyses:
        return None

    content = render_analyses_file(analyses, week_start, week_end)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        file_id = await gigachat.upload_file(tmp_path, "text/plain")
        if not file_id:
            return None
        result = await gigachat.send_prompt(COMMON_ISSUES_PROMPT, attachments=[file_id])
        if result in ("Ошибка от GigaChat API", "Ошибка выполнения запроса"):
            return None
        await save_analysis_to_cache(db, week_start, week_end, result)
        return result
    finally:
        os.unlink(tmp_path)