import csv
from pathlib import Path
from typing import List, Tuple


def read_questions(csv_path: Path) -> List[Tuple[str, str]]:
    questions = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "question"]:
            raise ValueError(f"Ожидались колонки ['id', 'question'], получены {reader.fieldnames}")
        for row in reader:
            q_id = row["id"].strip()
            question = row["question"].strip()
            if not q_id:
                continue
            questions.append((q_id, question))
    return questions


def write_submission(answers: List[Tuple[str, str]], output_path: Path) -> None:
    seen_ids = set()
    for q_id, _ in answers:
        if q_id in seen_ids:
            raise ValueError(f"Дублирующийся ID: {q_id}")
        seen_ids.add(q_id)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "answer"])
        for q_id, answer in answers:
            if not answer or not answer.strip():
                answer = "В предоставленных материалах этого нет / для ответа нужен специалист."
            if len(answer) > 10000:
                answer = answer[:10000]
            writer.writerow([q_id, answer.strip()])


def parse_conversation(text: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    Парсит историю диалога из текста вопроса.
    Возвращает (history, current_question).
    History — список (role, content), где role — 'user' или 'assistant'.
    """
    lines = text.strip().split("\n")
    history = []
    current_question = text
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("User:") or line_stripped.startswith("Пользователь:"):
            content = line_stripped.split(":", 1)[1].strip()
            history.append(("user", content))
        elif line_stripped.startswith("Assistant:") or line_stripped.startswith("Ассистент:"):
            content = line_stripped.split(":", 1)[1].strip()
            history.append(("assistant", content))
        elif line_stripped.startswith("Human:") or line_stripped.startswith("Человек:"):
            content = line_stripped.split(":", 1)[1].strip()
            history.append(("user", content))
        elif line_stripped.startswith("AI:") or line_stripped.startswith("ИИ:"):
            content = line_stripped.split(":", 1)[1].strip()
            history.append(("assistant", content))
    
    if history:
        current_question = lines[-1].strip()
        for prefix in ["User:", "Пользователь:", "Human:", "Человек:"]:
            if current_question.startswith(prefix):
                current_question = current_question.split(":", 1)[1].strip()
                break
    
    return history, current_question