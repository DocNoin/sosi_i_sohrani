#!/usr/bin/env python3
"""
Hack&Change 2026 — Точка входа RAG-решения
Читает questions.csv, отвечает через RAG, пишет submission.csv
"""

import sys
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import load_config, read_questions, write_submission, parse_conversation
from rag.knowledge import load_knowledge_base
from rag.retrieval import HybridRetriever
from rag.rerank import Reranker
from rag.generation import Generator


def process_question(
    q_id: str,
    question: str,
    retriever: HybridRetriever,
    reranker: Reranker,
    generator: Generator,
    config,
) -> tuple:
    import time

    try:
        t0 = time.perf_counter()
        history, current_q = parse_conversation(question)

        t1 = time.perf_counter()
        candidates = retriever.search(current_q, config.top_k)
        t2 = time.perf_counter()

        if config.use_reranker and candidates:
            reranked = reranker.rerank(current_q, candidates, config.reranker_top_n)
        else:
            from rag.rerank import RerankResult
            reranked = [RerankResult(chunk=r.chunk, score=r.score) for r in candidates[:config.reranker_top_n]]
        t3 = time.perf_counter()

        result = generator.generate(current_q, history, reranked)
        t4 = time.perf_counter()

        print(
            f"[{q_id}] retrieve={t2-t1:.2f}s rerank={t3-t2:.2f}s generate={t4-t3:.2f}s total={t4-t0:.2f}s",
            file=sys.stderr,
            flush=True,
        )

        return q_id, result.answer

    except Exception as e:
        print(f"Ошибка обработки вопроса {q_id}: {e}", file=sys.stderr)
        traceback.print_exc()
        return q_id, "В предоставленных материалах этого нет / для ответа нужен специалист."


def main():
    try:
        print("Starting main...", file=sys.stderr, flush=True)
        config = load_config()
        print(f"Конфиг загружен: model={config.llm_model}, data_dir={config.data_dir}", file=sys.stderr, flush=True)
        
        questions_path = config.data_dir / "questions.csv"
        if not questions_path.exists():
            raise FileNotFoundError(f"questions.csv не найден по пути {questions_path}")
        
        questions = read_questions(questions_path)
        print(f"Загружено {len(questions)} вопросов", file=sys.stderr, flush=True)
        
        print("Загрузка базы знаний...", file=sys.stderr, flush=True)
        kb_chunks = load_knowledge_base(
            config.data_dir,
            config.chunker_chunk_size,
            config.chunker_overlap
        )
        print(f"Загружено {len(kb_chunks)} фрагментов из базы знаний", file=sys.stderr, flush=True)
        
        print("Построение индексов...", file=sys.stderr, flush=True)
        retriever = HybridRetriever(config)
        retriever.build(kb_chunks)
        
        reranker = Reranker(config)
        generator = Generator(config)
        
        print("Обработка вопросов...", file=sys.stderr, flush=True)
        answers = []
        
        if config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                future_to_qid = {
                    executor.submit(
                        process_question, q_id, question, retriever, reranker, generator, config
                    ): q_id
                    for q_id, question in questions
                }
                
                for future in as_completed(future_to_qid):
                    q_id, answer = future.result()
                    answers.append((q_id, answer))
                    print(f"Завершён {q_id}", file=sys.stderr, flush=True)
        else:
            for q_id, question in questions:
                q_id, answer = process_question(q_id, question, retriever, reranker, generator, config)
                answers.append((q_id, answer))
                print(f"Завершён {q_id}", file=sys.stderr, flush=True)
        
        answers.sort(key=lambda x: x[0])
        
        output_path = Path.cwd() / "submission.csv"
        write_submission(answers, output_path)
        print(f"Записано {len(answers)} ответов в {output_path}", file=sys.stderr, flush=True)
        
    except Exception as e:
        print(f"Критическая ошибка: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()