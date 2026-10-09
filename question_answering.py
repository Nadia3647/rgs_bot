import logging
import os
import tempfile
from typing import List

from services.gigachat_service import GigaChatClient
from utils.audio_processor import async_remove_files, async_write_binary_file

logger = logging.getLogger(__name__)

NOT_FOUND_MARKER = "информация не найдена"

QUESTION_PROMPT = """Пользователь задал вопрос: "{question}"

Проанализируй данную часть аудиозаписи собрания по {region} и:
1. Если есть ответ, приведи точную цитату и объяснение. В ответе не упоминай слово «аудиозапись», пиши просто ответ по теме.
2. Если ответа нет, верни "Информация не найдена".
"""


async def answer_question_from_audio(
    question: str,
    audio_chunks: List[bytes],
    gigachat: GigaChatClient,
    region: str,
) -> str:
    """Ищет ответ на вопрос по аудиофрагментам собрания, хранящимся в БД."""
    answers = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, chunk_data in enumerate(audio_chunks):
            if not chunk_data:
                continue

            chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
            await async_write_binary_file(chunk_path, chunk_data)

            try:
                file_id = await gigachat.upload_file(chunk_path, "audio/mp3")
                if not file_id:
                    continue
                prompt = QUESTION_PROMPT.format(question=question, region=region)
                response = await gigachat.send_prompt(prompt, attachments=[file_id])
                if NOT_FOUND_MARKER not in response.lower():
                    answers.append(response.strip())
            except Exception as e:
                logger.error("Ошибка обработки фрагмента %s: %s", i, e)
            finally:
                await async_remove_files([chunk_path])

    if not answers:
        return f"В материалах собрания по {region} не найдено информации по вашему вопросу."

    return f"На основе анализа собрания {region}:\n\n" + "\n".join(answers)