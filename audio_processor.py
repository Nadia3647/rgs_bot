import asyncio
import logging
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from config.settings import CHUNKS_DIR, MAX_FILE_SIZE_MB, MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)

io_executor = ThreadPoolExecutor(max_workers=4)

DOWNLOAD_FORMATS = ["bestaudio/best", "bestaudio", "dash_sep-11", "mp4"]
MIN_VALID_FILE_SIZE = 10 * 1024  # файлы меньше считаются повреждёнными


async def async_download_audio(vk_url: str, output_dir: str) -> Optional[str]:
    """Скачивает аудиодорожку видео из VK в формате mp3."""

    def _download() -> Optional[str]:
        audio_path = os.path.join(output_dir, "audio.mp3")
        cookies_args = ["--cookies", "cookies.txt"] if os.path.exists("cookies.txt") else []

        for attempt in range(1, MAX_RETRIES + 1):
            for fmt in DOWNLOAD_FORMATS:
                cmd = [
                    "yt-dlp", *cookies_args, "-f", fmt,
                    "--extract-audio", "--audio-format", "mp3",
                    "-o", audio_path, vk_url,
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > MIN_VALID_FILE_SIZE:
                        return audio_path
                except subprocess.CalledProcessError as e:
                    logger.warning("Формат %s, попытка %s: %s", fmt, attempt, (e.stderr or "")[:200])
                except subprocess.TimeoutExpired:
                    logger.warning("Таймаут скачивания (формат %s, попытка %s)", fmt, attempt)

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        logger.error("Не удалось скачать аудио: %s", vk_url)
        return None

    return await asyncio.get_running_loop().run_in_executor(io_executor, _download)


def _get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return float(result.stdout.strip())


async def async_split_audio_ffmpeg(
    input_file: str,
    max_size_mb: float = MAX_FILE_SIZE_MB,
    output_dir: str = str(CHUNKS_DIR),
    overlap_seconds: int = 5,
) -> List[str]:
    """Режет аудио на фрагменты не больше max_size_mb с перекрытием overlap_seconds."""

    def _split() -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        try:
            total_duration = _get_duration(input_file)
        except Exception as e:
            logger.error("Не удалось определить длительность: %s", e)
            return []

        # длительность фрагмента вычисляется из среднего размера секунды записи
        bytes_per_second = os.path.getsize(input_file) / total_duration
        chunk_duration = int(max_size_mb * 1024 * 1024 / bytes_per_second)
        step = chunk_duration - overlap_seconds
        if step <= 0:
            raise ValueError("Перекрытие больше длительности фрагмента")

        num_chunks = math.ceil(total_duration / step)
        chunk_files = []

        for i in range(num_chunks):
            start = max(0, i * step - (overlap_seconds if i > 0 else 0))
            output_file = os.path.join(output_dir, f"chunk_{i}.mp3")
            cmd = [
                "ffmpeg", "-y", "-i", input_file,
                "-ss", str(start), "-t", str(chunk_duration),
                "-c", "copy", output_file,  # без перекодирования: быстро и без потери качества
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                logger.error("Не удалось создать фрагмент %s: %s", i, result.stderr[:300])
                continue
            chunk_files.append(output_file)

        logger.info("Создано %s фрагментов из %s", len(chunk_files), num_chunks)
        return chunk_files

    return await asyncio.get_running_loop().run_in_executor(io_executor, _split)


# --- вспомогательные асинхронные операции с файлами ---

async def async_write_file(path: str, content: str) -> None:
    def _write():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    await asyncio.get_running_loop().run_in_executor(io_executor, _write)


async def async_write_binary_file(path: str, data: bytes) -> None:
    def _write():
        with open(path, "wb") as f:
            f.write(data)
    await asyncio.get_running_loop().run_in_executor(io_executor, _write)


async def async_read_file(path: str) -> bytes:
    def _read():
        with open(path, "rb") as f:
            return f.read()
    return await asyncio.get_running_loop().run_in_executor(io_executor, _read)


async def async_remove_files(paths: List[str]) -> None:
    def _remove_all():
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
    await asyncio.get_running_loop().run_in_executor(io_executor, _remove_all)