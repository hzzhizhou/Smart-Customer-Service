"""
评估层：RAGAS评估+稳定化+指标监控
规避：评估结果波动、指标无监控问题
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 以 `python evaluation/service.py` 直接运行时，sys.path[0] 是 evaluation/ 而不是项目根，
# 会导致下面 `import retrieval.*` 失败，所以先把项目根塞进来
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ragas import aevaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)
from ragas.run_config import RunConfig
from datasets import Dataset

import retrieval.service as retrieval_service_module
from retrieval.reranker import CrossEncoderReranker
from utils.thread_pool_manager import init_thread_pools
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from retrieval.service import RetrievalService
from generation.service import AnswerGenerator
from infrastructure.chat_history_factory import init_chat_history
from config.settings import (
    EVAL_RUNS, EVAL_THRESHOLDS, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED,
    LOCAL_EMBEDDING_MODEL,
)
from utils.llm_factory import create_llm
from logs.log_config import evaluation_layer_log as log

EVAL_SET_PATH = BASE_DIR / "evaluation" / "eval_set.json"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# 判分并发：单题内四个指标会一起进这个池子，4 与 DashScope 的限流比较匹配
JUDGE_MAX_WORKERS = 4
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-v4.1-flash")
# 判分温度：判分是分类任务，要的是确定性而非发散。1.0 只是当初给「只接受 1.0 的
# kimi-k3」留的兼容值；已实测 qwen3.8-max 接受 0.01，故改回 0.01 压掉判分抖动。
JUDGE_TEMPERATURE = 0.01
# 判分随机种子：与温度一起固定，让同一份输入重跑得到同样的分（实验可复现）
JUDGE_SEED = 42
# 送判 context 上限：判分只看检索前 3 条（省 token 口径，双臂一致，历史口径不变）
JUDGE_CONTEXT_TOP_K = 3


class RAGEvaluator:
    """企业级RAG评估器：稳定化指标、监控告警、历史记录"""

    def __init__(self, vector_store, retrieval_service, answer_generator):
        self.vector_store = vector_store
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator

        # 判分模型走 OpenAI 兼容端点，参数约束比原生接口严。包一层 LangchainLLMWrapper
        # 并打开 bypass，避免 ragas 内部擅自改参数把请求打回 400：
        #   bypass_temperature：否则 ragas 会把 temperature 覆写成 0.01（见 llms/base.py）
        #   bypass_n：否则 answer_relevancy 的 strictness=3 会以 n=3 单次请求发出，
        #             有些端点只允许 n=1；bypass 后改成同 prompt 发 3 次，语义等价
        self.ragas_llm = LangchainLLMWrapper(
            create_llm(model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE, seed=JUDGE_SEED),
            bypass_temperature=True,
            bypass_n=True,
        )

        # 获取嵌入模型（用于 answer_relevancy）
        embedding_model = vector_store.embedding_model
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
        # answer_relevancy 默认 strictness=3：生成 3 个候选问题再算相似度取平均，
        # 是四个指标里最贵的一项。降到 1 省掉 2/3 调用（代价是该项抖动略增）。
        answer_relevancy.strictness = 1
        # 最近一次 _generate_test_data_async 的平均单题检索耗时，供对比表展示
        self.last_retrieval_avg = 0.0

    async def _generate_test_data_async(self,
                                        questions: List[str],
                                        ground_truths: Optional[List[str]] = None) -> Dataset:
        """
        逐题检索+生成数据集。

        顺序跑而不是 asyncio.gather 全并发：一是终端能看到 `[i/N]` 逐题进度
        （全并发只在最后一次性返回，中间完全黑箱），二是避免一口气打出几十个
        请求触发 DashScope 限流。
        """
        results = []
        retrieval_secs = []
        for i, q in enumerate(questions, 1):
            gt = ground_truths[i - 1] if ground_truths else None
            t0 = time.time()
            # 执行检索（注意：retrieve 是异步方法）
            docs, retriever_type, _ = await self.retrieval_service.retrieve(
                question=q,
                use_context=False,   # 评估时不使用对话历史改写
                use_hyde=False,
                use_multi=False,
            )
            retrieval_sec = time.time() - t0
            # 生成答案仍用全部检索结果，不牺牲答案质量
            answer = await self.answer_generator.generate(q, docs, session_id=None)
            # 送判的 context 只取前 K 条：这直接决定 context_precision 的调用次数与总 token
            judge_contexts = [d.page_content for d in docs[:JUDGE_CONTEXT_TOP_K]]
            results.append({
                "question": q,
                "answer": answer,
                "contexts": judge_contexts,
                "ground_truth": gt if gt else "",
            })
            retrieval_secs.append(retrieval_sec)
            print(f"      检索+生成 [{i}/{len(questions)}] {q[:26]} | "
                  f"检索 {retrieval_sec:.2f}s | 检索到 {len(docs)} 条 → "
                  f"送判 {len(judge_contexts)} 条 | 答复 {len(answer)} 字", flush=True)

        data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
        }
        if ground_truths:
            data["ground_truth"] = [r["ground_truth"] for r in results]

        self.last_retrieval_avg = (
            sum(retrieval_secs) / len(retrieval_secs) if retrieval_secs else 0.0
        )
        return Dataset.from_dict(data)

    async def _score_dataset(self, dataset: Dataset) -> List[Dict[str, Optional[float]]]:
        """跑四个指标，返回与 dataset 行同序的逐题分数。

        均值统一走 _mean()，不要用 result.to_pandas()[col].iloc[0] —— 那是拿
        第一题的分数冒充整体均值。raise_exceptions=False 时失败指标为 NaN，
        这里置成 None 由 _mean() 跳过，不污染均值。
        """
        result = await aevaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            raise_exceptions=False,
            run_config=RunConfig(max_workers=JUDGE_MAX_WORKERS),
            show_progress=True,
        )
        rows = []
        for row in (result.scores or []):
            one = {}
            for name in METRIC_NAMES:
                v = row.get(name)
                # NaN 不等于自身，用它把失败项筛掉
                one[name] = float(v) if isinstance(v, (int, float)) and v == v else None
            rows.append(one)
        return rows

    @staticmethod
    def _mean(rows: List[Dict[str, Optional[float]]],
              metric: str,
              indices: Optional[List[int]] = None) -> Optional[float]:
        """对指定题号子集求某指标均值；无有效值时返回 None（显示 n/a）。"""
        picked = rows if indices is None else [rows[i] for i in indices]
        vals = [r.get(metric) for r in picked if r.get(metric) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    async def evaluate_async(self,
                            questions: List[str],
                            ground_truths: Optional[List[str]] = None,
                            runs: int = None) -> Dict[str, float]:
        if runs is None:
            runs = EVAL_RUNS
        # 只运行一次（多次无意义，白烧 token）
        if runs != 1:
            log.warning(f"RAGAS 评估建议 runs=1，当前 runs={runs}，将只运行一次")
        # 生成数据集
        dataset = await self._generate_test_data_async(questions, ground_truths)
        rows = await self._score_dataset(dataset)
        avg_results = {name: self._mean(rows, name) for name in METRIC_NAMES}
        log.info(f"评估完成，结果：{avg_results}")
        return avg_results

    def evaluate(self,
                 questions: List[str],
                 ground_truths: Optional[List[str]] = None,
                 runs: int = None) -> Dict[str, float]:
        """
        同步包装器（供非异步环境调用）
        """
        return asyncio.run(self.evaluate_async(questions, ground_truths, runs))

    def evaluate_with_report(self,
                             questions: List[str],
                             ground_truths: Optional[List[str]] = None,
                             runs: int = None) -> Dict[str, Any]:
        """
        执行评估并生成详细报告（同步）
        """
        avg_results = self.evaluate(questions, ground_truths, runs)

        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_runs": runs if runs else EVAL_RUNS,
            "num_questions": len(questions),
            "metrics": avg_results,
            "thresholds": EVAL_THRESHOLDS,
            "alerts": []
        }

        for metric, threshold in EVAL_THRESHOLDS.items():
            value = avg_results.get(metric)
            if value is not None and value < threshold:
                report["alerts"].append({
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "suggestion": f"建议优化{metric}相关组件"
                })
        return report

    # ---------- 重排序效果对比 ----------

    def _set_rerank(self, enabled: bool) -> None:
        """就地切换重排：模块全局（_merge_results 读它）与实例上的 reranker 一起换。

        CrossEncoderReranker 首次构造要读约 1.1GB 权重（实测约 20s），只在开启臂付一次。
        """
        retrieval_service_module.RERANK_MODE = "cross_encoder" if enabled else "none"
        self.retrieval_service.reranker = CrossEncoderReranker() if enabled else None

    async def compare_rerank(self,
                             items: List[dict],
                             limit: Optional[int] = None) -> Dict[str, Any]:
        """关闭 / 开启重排各评一轮（runs=1），返回两套整体指标并打印对比表。

        items 为 eval_set.json 的条目（含 question / reference / answerable）。
        每臂只跑一轮：重复跑对「重排有没有用」没有增量信息，只是白烧 token。
        """
        if limit:
            items = items[:limit]
        questions = [it["question"] for it in items]
        ground_truths = [it["reference"] for it in items]
        answerable_idx = [i for i, it in enumerate(items) if it.get("answerable", True)]
        hard_idx = [i for i in range(len(items)) if i not in answerable_idx]

        print(f"\n重排序效果对比：共 {len(items)} 题（可答 {len(answerable_idx)} 题，"
              f"无答案 {len(hard_idx)} 题），每臂只跑 1 轮", flush=True)

        arms = {}
        for label, enabled in (("关闭重排", False), ("开启重排", True)):
            print(f"\n===== 臂：{label} =====", flush=True)
            t0 = time.time()
            self._set_rerank(enabled)
            dataset = await self._generate_test_data_async(questions, ground_truths)
            gen_sec = time.time() - t0
            t1 = time.time()
            rows = await self._score_dataset(dataset)
            print(f"[{label}] 检索+生成 {gen_sec:.0f}s，判分 {time.time() - t1:.0f}s，"
                  f"平均检索 {self.last_retrieval_avg:.2f}s/题", flush=True)
            arms[label] = {
                "rows": rows,
                "overall": {m: self._mean(rows, m, answerable_idx) for m in METRIC_NAMES},
                "unanswerable": {m: self._mean(rows, m, hard_idx) for m in METRIC_NAMES},
                "retrieval_avg": self.last_retrieval_avg,
                "gen_sec": gen_sec,
            }

        self._set_rerank(True)   # 还原成项目默认配置，避免影响后续调用
        self._print_compare(arms, len(items), len(answerable_idx))
        return arms

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    @classmethod
    def _print_compare(cls, arms: Dict[str, Any], n_total: int, n_answerable: int) -> None:
        """打印两臂整体指标对比表（这是看结果的地方）。"""
        left, right = "关闭重排", "开启重排"
        print("\n" + "=" * 78)
        print(f"重排序效果对比 · 整体均值（每臂 1 轮，共 {n_total} 题，"
              f"其中可答 {n_answerable} 题）")
        print("=" * 78)
        print(f"{'指标':<24}{left:>16}{right:>16}{'差值':>14}")
        print("-" * 78)
        for m in METRIC_NAMES:
            a = arms[left]["overall"][m]
            b = arms[right]["overall"][m]
            diff = "n/a" if a is None or b is None else f"{b - a:+.4f}"
            print(f"{m:<24}{cls._fmt(a):>16}{cls._fmt(b):>16}{diff:>14}")
        print("-" * 78)
        print(f"{'平均检索耗时':<24}"
              f"{arms[left]['retrieval_avg']:.2f}s/题".rjust(16) +
              f"{arms[right]['retrieval_avg']:.2f}s/题".rjust(16))
        print("=" * 78)
        print("注：差值 = 开启重排 - 关闭重排，正值为重排更优；"
              "context_precision / context_recall 只在可答题上统计。")
        print(f"注：送判 context 上限为前 {JUDGE_CONTEXT_TOP_K} 条，"
              f"answer_relevancy.strictness=1（省 token 口径，两臂一致）。")
        if arms[left]["unanswerable"]["faithfulness"] is not None:
            print("无答案题（应拒答）faithfulness："
                  f"关闭重排 {cls._fmt(arms[left]['unanswerable']['faithfulness'])}，"
                  f"开启重排 {cls._fmt(arms[right]['unanswerable']['faithfulness'])}"
                  "（衡量有没有编造知识库里没有的内容）")


if __name__ == '__main__':
    """重排序效果对比入口：对评测集各跑一轮「关闭重排 / 开启重排」，打印整体指标对比。

    用法（在 RagServerSystem 目录下）：
        python evaluation/service.py              # 跑全部题目
        python evaluation/service.py --limit 10   # 只跑前 10 题（token 紧张时用）
    """

    def parse_args():
        parser = argparse.ArgumentParser(description="重排序前后 RAGAS 整体指标对比")
        parser.add_argument("--limit", type=int, default=None,
                            help="只跑前 N 题（token 紧张时用）")
        return parser.parse_args()

    async def main():
        args = parse_args()
        items = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
        print(f"评测集 {EVAL_SET_PATH.name}：共 {len(items)} 题")

        print("正在初始化组件（首次要加载向量库，开启臂还要读重排模型，请稍候）...")
        vector_store = ChromaVector()
        llm = create_llm(model=LLM_MODEL, temperature=LLM_TEMPERATURE, seed=LLM_SEED)
        init_thread_pools()
        retrieval_service = RetrievalService(vector_store, llm, None)
        answer_generator = AnswerGenerator(llm)
        evaluator = RAGEvaluator(vector_store, retrieval_service, answer_generator)

        started = time.time()
        await evaluator.compare_rerank(items, limit=args.limit)
        print(f"\n两臂合计耗时 {time.time() - started:.0f}s")

    asyncio.run(main())