import asyncio
import logging
import time
from typing import List

from config.prompts import CHUNK_PROMPT_BASE, CHUNK_PROMPT_WITH_LEASING, FINAL_REPORT_PROMPT
from config.settings import GROUPS_WITH_NOTIFICATIONS, MAX_RETRIES, RETRY_DELAY
from utils.audio_processor import async_remove_file, async_write_file

logger = logging.getLogger(__name__)


async def analyze_chunk(gigachat, chunk_path: str, part_number: int, group_code: str) -> str:
    """Анализирует один аудиофрагмент. Возвращает текст ответа или маркер ошибки."""
    base_prompt = CHUNK_PROMPT_WITH_LEASING if group_code in GROUPS_WITH_NOTIFICATIONS else CHUNK_PROMPT_BASE
    prompt = f"{base_prompt}\nЧасть {part_number}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            file_id = await gigachat.upload_file(chunk_path, "audio/mp3")
            if not file_id:
                raise RuntimeError("Ошибка загрузки файла")

            response = await gigachat.send_prompt(prompt, attachments=[file_id])
            if not response or not response.strip():
                raise RuntimeError("Пустой ответ от GigaChat")
            return response
        except Exception as e:
            logger.error("Часть %s, попытка %s: %s", part_number, attempt, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

    return f"[Ошибка обработки части {part_number}]"


async def build_final_report(chunk_files: List[str], gigachat, group_code: str) -> str:
    """Map-reduce: анализ каждого фрагмента, затем сведение результатов в единый отчёт."""
    partial_results = [
        await analyze_chunk(gigachat, path, idx + 1, group_code)
        for idx, path in enumerate(chunk_files)
    ]

    valid = [r for r in partial_results if "Ошибка обработки части" not in r]
    if not valid:
        raise RuntimeError("Не удалось обработать ни один фрагмент записи")

    # промежуточный результат передаётся в модель как файл-вложение:
    # так обходится ограничение на длину промпта
    intermediate_path = f"intermediate_{group_code}_{int(time.time())}.txt"
    await async_write_file(intermediate_path, "\n\n".join(valid))

    try:
        file_id = await gigachat.upload_file(intermediate_path, "text/plain")
        if not file_id:
            raise RuntimeError("Не удалось загрузить промежуточный файл")

        report = await gigachat.send_prompt(FINAL_REPORT_PROMPT, attachments=[file_id])
        if not report or not report.strip():
            raise RuntimeError("Пустой итоговый отчёт")
        return report
    finally:
        await async_remove_file(intermediate_path)