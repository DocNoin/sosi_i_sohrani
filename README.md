# МТС EnergyTool — ИИ-консультант для сайта

Решение для кейса **МТС True Tech Hack&Change 2026**: ИИ-консультант, который помогает клиентам подбирать продукты EnergyTool, отвечает на вопросы по энергосбережению, учёту электроэнергии и тарифам, собирает заявки для менеджеров.

---

# Архитектура решения

```
┌─────────────────────────────────────────────────────────────────┐
│                      Web Interface (FastAPI)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│  │ Chat UI     │  │ Session Mgmt │  │ Lead Collection       │   │
│  │ + Debug     │  │ (history,    │  │ (ФИО, телефон,        │   │
│  │ mode        │  │  context,    │  │  компания, резюме)    │   │
│  │             │  │  reset)      │  │                       │   │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘   │
└─────────┼────────────────┼───────────────────────┼──────────────┘
          │                │                       │
          ▼                ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│  │ Hybrid      │  │ Reranker     │  │ Generator             │   │
│  │ Retriever   │  │ (optional)   │  │ (Qwen3.8-27B /        │   │ 
│  │ (BM25 +     │  │ Qwen3        │  │  local qwen3-8b)      │   │
│  │  Vector)    │  │ Reranker 8B) │  │                       │   │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘   │
└─────────┼────────────────┼───────────────────────┼──────────────┘
          │                │                       │
          ▼                ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Knowledge Base                              │
│  • knowledge-base.zip (Markdown/PDF/HTML)                       │
│  • Чанкинг (512 слов, overlap 100)                              │
│  • Эмбеддинги: BAAI/bge-m3                                      │
│  • Кэширование эмбеддингов (.rag_cache)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Основные компоненты

| Компонент            | Описание                                                                               |
|----------------------|----------------------------------------------------------------------------------------|
| `solution.py`        | Точка входа для Stage 1 (автопроверка): читает `questions.csv`, пишет `submission.csv` |
| `app/main.py`        | FastAPI веб-сервер с чат-интерфейсом                                                   |
| `app/session.py`     | Управление диалоговыми сессиями (контекст, история, сброс, лиды)                       |
| `app/rag_service.py` | Обёртка над RAG-пайплайном для веба (консультант-поведение, уточнения, специалисты)    |
| `rag/knowledge.py`   | Загрузка и чанкинг базы знаний из `knowledge-base.zip`                                 |
| `rag/retrieval.py`   | Гибридный поиск (BM25 + Vector, RRF), кэш эмбеддингов                                  |
| `rag/rerank.py`      | Реранкинг через `Qwen3 Reranker 8B` (опционально)                                      |
| `rag/generation.py`  | Генерация ответов через LLM API                                                        |
| `utils/env.py`       | Конфигурация через переменные окружения                                                |
| `utils/csv_io.py`    | Чтение вопросов и запись `submission.csv`                                              |

---

## Быстрый старт

### 1. Локальная разработка (Python)

```bash
# Клонирование и переход в папку
cd olimp

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения (пример для LM Studio)
export DATA_DIR="./data"
export LLM_BASE_URL="http://localhost:1234/v1"
export LLM_API_KEY="lm-studio"
export LLM_MODEL="qwen3-8b"
export EMB_MODEL="text-embedding-nomic-embed-text-v2-moe"
export USE_RERANKER="false"
export MAX_WORKERS="1"

# Запуск веб-интерфейса
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Откройте http://localhost:8000
```

## 2. Запуск через Docker

```bash
# Подготовка .env файла (скопируйте из .env.example)
cp .env.example .env
# Отредактируйте .env — укажите ваши LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMB_MODEL

# Сборка и запуск веб-интерфейса
docker compose up --build energytool-chat

# Откройте http://localhost:8000
```

## 3. Stage 1 сабмит (генерация submission.csv)

```bash
# Через Docker (профиль submit)
docker compose --profile submit up --build energytool-submit

# Результат появится в ./output/submission.csv
# Или локально:
python solution.py
# Результат в ./submission.csv
```

---

## Переменные окружения

| Переменная | Описание | По умолчанию | Обязательна |
|------------|----------|--------------|-------------|
| `DATA_DIR` | Путь к папке с `questions.csv` и `knowledge-base.zip` | `./data` | Да (на платформе задаётся организаторами) |
| `LLM_BASE_URL` | Базовый URL LLM API (OpenAI-совместимый) | — | **Да** |
| `LLM_API_KEY` | API ключ для LLM | — | **Да** |
| `LLM_MODEL` | Имя модели генерации (например, `Qwen3.8-27B` на платформе) | — | **Да** |
| `OPENAI_BASE_URL` | Альтернативный URL (если нет LLM_BASE_URL) | — | Нет |
| `OPENAI_API_KEY` | Альтернативный ключ | — | Нет |
| `OPENAI_MODEL` / `GEN_MODEL` | Альтернативные имена модели генерации | — | Нет |
| `EMB_MODEL` | Модель эмбеддингов (на платформе `BAAI/bge-m3`) | `BAAI/bge-m3` | Да |
| `TEMPERATURE` | Температура генерации | `0.1` | Нет |
| `MAX_TOKENS` | Макс. токенов в ответе | `512` | Нет |
| `MAX_WORKERS` | Параллельность для Stage 1 | `1` | Нет |
| `TOP_K` | Количество кандидатов из ретривера | `10` | Нет |
| `CHUNKER_CHUNK_SIZE` | Размер чанка (слова) | `512` | Нет |
| `CHUNKER_OVERLAP` | Перекрытие чанков (слова) | `100` | Нет |
| `RERANKER_MODEL` | Модель реранкера | `Qwen3-Reranker-8B` | Нет |
| `RERANKER_TOP_N` | Топ-N после реранкинга | `3` | Нет |
| `USE_RERANKER` | Включить реранкер (`true`/`false`) | `true` | Нет |
| `REQUEST_TIMEOUT` | Таймаут запросов к LLM (сек) | `180` | Нет |
| `MAX_RETRIES` | Повторные попытки при ошибках | `3` | Нет |

> **На платформе** организаторы предоставят свои значения для `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (`Qwen3.8-27B`), `EMB_MODEL` (`BAAI/bge-m3`), `RERANKER_MODEL` (`Qwen3 Reranker 8B`). Ваш код должен читать их из окружения — **не хардкодить**.

---

## Возможности чат-интерфейса

- **Многошаговый диалог** — контекст сохраняется в рамках сессии
- **Режим отладки** — показывает время поиска/реранка/генерации, найденные фрагменты БЗ
- **Сброс контекста** — кнопка «Новый диалог» полностью очищает историю
- **Сбор заявки** — при готовности клиента предлагает ввести ФИО, телефон, компанию
- **Резюме для менеджера** — автоматическое формирование краткого саммари диалога
- **История сессий** — все диалоги (в т.ч. без заявки) сохраняются в памяти

### Режим отладки
Включается чекбоксом в правом верхнем углу. Показывает:
- Время: retrieve / rerank / generate / total
- Найденные фрагменты (до 5) с scores и источниками
- После реранкинга (top-N)
- Токены prompt/completion

---

## Структура проекта

```
olimp/
├── solution.py              # Stage 1 entry point
├── requirements.txt         # Зависимости
├── Dockerfile               # Docker образ
├── docker-compose.yml       # Оркестрация (chat + submit)
├── .dockerignore            # Исключения для Docker
├── .env.example             # Пример переменных окружения
├── README.md                # Этот файл
├── data/                    # Входные данные (questions.csv, knowledge-base.zip)
├── rag/                     # RAG пайплайн
│   ├── __init__.py
│   ├── knowledge.py
│   ├── retrieval.py
│   ├── rerank.py
│   └── generation.py
├── utils/                   # Утилиты
│   ├── __init__.py
│   ├── env.py
│   └── csv_io.py
├── app/                     # Веб-приложение
│   ├── __init__.py
│   ├── main.py              # FastAPI
│   ├── schemas.py           # Pydantic модели
│   ├── session.py           # Сессии
│   └── rag_service.py       # RAG для веба
└── templates/
    └── chat.html            # Чат UI с debug режимом
```

---

## 📝 Лицензия