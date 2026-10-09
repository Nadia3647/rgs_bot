import base64
import logging
import os
import time
import uuid
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_BASE = "https://gigachat.devices.sberbank.ru/api/v1"
TOKEN_SAFETY_MARGIN = 60  # секунд до истечения, когда токен считается устаревшим


class GigaChatClient:
    def __init__(self, config: dict):
        self.config = config
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0.0

    # --- авторизация ---

    async def _get_new_token(self) -> bool:
        credentials = f"{self.config['client_id']}:{self.config['client_secret']}"
        auth_key = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {auth_key}",
        }
        data = {"scope": self.config.get("scope", "")}

        try:
            # verify=False нужен при использовании корпоративного сертификата Минцифры;
            # в продакшене лучше подключить корневой сертификат вместо отключения проверки
            async with httpx.AsyncClient(timeout=100.0, verify=False) as client:
                response = await client.post(AUTH_URL, headers=headers, data=data)
        except httpx.RequestError as e:
            logger.error("Сетевая ошибка при получении токена: %s", e)
            return False

        if response.status_code != 200:
            logger.error("Ошибка авторизации (HTTP %s): %s", response.status_code, response.text)
            return False

        payload = response.json()
        token = payload.get("access_token")
        if not token:
            logger.error("В ответе нет access_token: %s", payload)
            return False

        expires_in = payload.get("expires_in", 1800)
        self.access_token = token
        self.token_expires_at = time.time() + expires_in - TOKEN_SAFETY_MARGIN
        return True

    async def _ensure_valid_token(self) -> None:
        if not self.access_token or time.time() >= self.token_expires_at:
            if not await self._get_new_token():
                raise RuntimeError("Не удалось обновить токен GigaChat")

    # --- загрузка файлов ---

    async def upload_file(self, file_path: str, mime_type: str, max_attempts: int = 3) -> Optional[str]:
        """Загружает файл в хранилище GigaChat и возвращает его id."""
        await self._ensure_valid_token()
        url = f"{API_BASE}/files"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=300.0, verify=False) as client:
                    with open(file_path, "rb") as f:
                        files = {
                            "file": (os.path.basename(file_path), f, mime_type),
                            "purpose": (None, "general"),
                        }
                        response = await client.post(url, headers=headers, files=files)
            except Exception as e:
                logger.error("Ошибка загрузки файла (попытка %s): %s", attempt, e)
                continue

            if response.status_code == 200:
                return response.json().get("id")
            if response.status_code in (400, 401, 413):
                # повторять бессмысленно: неверный формат, авторизация или размер
                logger.error("Загрузка отклонена (HTTP %s): %s", response.status_code, response.text)
                return None
            logger.warning("Загрузка не удалась (HTTP %s), попытка %s", response.status_code, attempt)

        return None

    # --- генерация ---

    async def send_prompt(self, prompt: str, attachments: Optional[List[str]] = None) -> str:
        await self._ensure_valid_token()
        url = f"{API_BASE}/chat/completions"

        message = {"role": "user", "content": prompt}
        if attachments:
            message["attachments"] = attachments

        payload = {
            "model": self.config["model_name"],
            "messages": [message],
            "temperature": self.config["temperature"],
        }

        try:
            async with httpx.AsyncClient(timeout=200.0, verify=False) as client:
                headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
                response = await client.post(url, headers=headers, json=payload)

                # токен мог быть отозван: обновляем и повторяем запрос один раз
                if response.status_code == 401 and await self._get_new_token():
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            logger.error("Ошибка генерации (HTTP %s): %s", response.status_code, response.text)
            return "Ошибка от GigaChat API"
        except Exception as e:
            logger.error("Ошибка запроса к GigaChat: %s", e)
            return "Ошибка выполнения запроса"

    # --- эмбеддинги ---

    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        await self._ensure_valid_token()
        url = f"{API_BASE}/embeddings"
        headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
        payload = {"model": "Embeddings", "input": text}

        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        logger.error("Ошибка получения эмбеддинга: %s", response.text)
        return None