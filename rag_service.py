# services/rag_service.py
import json
import logging
import re
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """
На основе анализа оперативного совещания правительства региона предложи только релевантные продукты банка из списка.
Если релевантных больше трёх, выведи три самых релевантных. Если релевантных продуктов нет, не выдумывай и напиши, что для данных задач релевантные продукты не найдены.

Анализ совещания:
{analysis_text}

Список продуктов банка:
{products_text}

Требования к ответу:
1. Без общих фраз, без вступления и заключения, без подзаголовков.
2. Каждый продукт с новой строки, начиная с " - ".
3. Для каждого продукта: название, затем одно предложение о том, с какой задачей из обсуждения он связан и как именно поможет.
4. Используй только продукты из списка.
5. Без Markdown-разметки и символов "#".
"""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def find_similar_products(db, text_embedding: List[float], top_k: int = 5) -> List[dict]:
    """Возвращает top_k продуктов, ближайших к тексту по косинусной близости."""
    rows = await db.fetch("SELECT name, description, embedding FROM sber_products")
    query_vec = np.array(text_embedding, dtype=np.float32)

    scored = []
    for row in rows:
        product_vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
        scored.append((cosine_similarity(query_vec, product_vec), row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"name": r["name"], "description": r["description"]} for _, r in scored[:top_k]]


def split_report_into_blocks(report: str) -> List[Dict[str, Any]]:
    """
    Делит итоговый отчёт на смысловые блоки для отдельного поиска продуктов.

    Блоком считается пункт списка вместе с названием темы, к которой он относится.
    Раздел с финансовым сектором и блок рисков пропускаются: продукты подбираются
    по темам, решениям и возможностям.
    """
    blocks: List[Dict[str, Any]] = []
    section = None
    topic = None
    buffer: List[str] = []

    topic_pattern = re.compile(r"^\d+\.\s+(.+)$")
    dash_pattern = re.compile(r"^-\s+(.+)$")

    def flush():
        nonlocal buffer
        if buffer and section and topic:
            blocks.append({"id": len(blocks) + 1, "section": section,
                           "topic": topic, "content": " ".join(buffer).strip()})
        buffer = []

    section_markers = {
        "Основные обсуждаемые темы": "main_topics",
        "Ключевые решения и поручения": "key_decisions",
        "Упоминания финансового сектора/банков": "finance",
        "Риски и возможности для банка": "risks_opportunities",
    }

    for raw_line in report.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        marker = next((v for k, v in section_markers.items() if line.startswith(k)), None)
        if marker:
            flush()
            section, topic = marker, None
            continue

        if section == "finance":
            continue

        if section == "risks_opportunities":
            if "Потенциальные риски" in line:
                flush()
                topic = None
                continue
            if "Потенциальные возможности" in line:
                flush()
                topic = "Возможности"
                continue

        if section in ("main_topics", "key_decisions") and topic_pattern.match(line):
            flush()
            topic = topic_pattern.match(line).group(1).strip()
            continue

        dash = dash_pattern.match(line)
        if dash and section and topic:
            flush()
            buffer = [topic, dash.group(1)]
            continue

        if section and topic:
            buffer = buffer or [topic]
            buffer.append(line)

    flush()
    return blocks


async def recommend_products(report: str, gigachat, db) -> str:
    """Подбирает продукты для отчёта: поиск кандидатов по блокам, затем отбор моделью."""
    blocks = split_report_into_blocks(report) or [
        {"id": 1, "section": "full_text", "topic": "весь текст", "content": report}
    ]

    candidates: Dict[str, dict] = {}
    for block in blocks:
        if len(block["content"]) <= 10:
            continue
        embedding = await gigachat.get_embeddings(block["content"])
        if not embedding:
            continue
        for product in await find_similar_products(db, embedding, top_k=2):
            candidates[product["name"]] = product  # словарь убирает дубликаты

    if not candidates:
        return "Для данных задач релевантные продукты не найдены."

    products_text = "\n".join(
        f"{i}. {p['name']}: {p['description']}" for i, p in enumerate(candidates.values(), 1)
    )
    prompt = RAG_PROMPT_TEMPLATE.format(analysis_text=report, products_text=products_text)
    return await gigachat.send_prompt(prompt)