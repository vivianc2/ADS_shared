"""Orchestrator for static RPG scientist runs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas_rpg import StaticRPGQuestion

logger = logging.getLogger(__name__)


@dataclass
class RPGTurn:
    turn_index: int
    action_type: str
    action_content: str
    raw_llm_response: str = ""
    reasoning: str = ""
    scientist_memory: str = ""
    query_result: Optional[Dict[str, Any]] = None
    data_summary: Optional[str] = None
    answer_score: Optional[Dict[str, Any]] = None
    limit_event: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "action_type": self.action_type,
            "action_content": self.action_content,
            "raw_llm_response": self.raw_llm_response,
            "reasoning": self.reasoning,
            "scientist_memory": self.scientist_memory,
            "query_result": self.query_result,
            "data_summary": self.data_summary,
            "answer_score": self.answer_score,
            "limit_event": self.limit_event,
        }


@dataclass
class RPGExperimentResult:
    question: StaticRPGQuestion
    scientist_answer: str
    score: Dict[str, Any]
    turns: List[RPGTurn]
    total_queries: int
    max_queries: int
    resource_usage: Dict[str, Any]
    log_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": {
                "question_type": self.question.question_type,
                "question_text": self.question.question_text,
                "ground_truth": self.question.ground_truth,
                "answer_schema": self.question.answer_schema,
                "metadata": self.question.metadata,
            },
            "scientist_answer": self.scientist_answer,
            "score": self.score,
            "turns": [t.to_dict() for t in self.turns],
            "total_queries": self.total_queries,
            "max_queries": self.max_queries,
            "resource_usage": self.resource_usage,
            "log_path": self.log_path,
        }


@dataclass
class OrchestratorRPG:
    world_model: Any
    scientist: Any
    question: StaticRPGQuestion
    max_queries: int = 8
    max_turns: int = 12
    log_dir: str = "./rpg_agent_logs"
    scientist_model: Optional[str] = None
    dataset_file: Optional[str] = None

    _turns: List[RPGTurn] = field(default_factory=list, init=False)
    _experiment_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def run(self) -> RPGExperimentResult:
        logger.info("Starting RPG experiment %s", self._experiment_id)
        logger.info("Question: %s", self.question.question_text)
        if hasattr(self.world_model, "reset_sample_usage"):
            self.world_model.reset_sample_usage()

        world_info = self.world_model.get_world_info()
        self.scientist.initialize(world_info, self.question, self.max_queries)
        self.scientist.receive_system_message(self._budget_message())

        answer = ""
        score: Dict[str, Any] = {"success": False, "accepted": False, "error": "no answer"}

        for turn_idx in range(1, self.max_turns + 1):
            action = self.scientist.get_next_action()
            action_type = action.get("type", "give_up")
            content = action.get("content", "")

            if action_type == "query":
                if self._successful_queries_used() >= self.max_queries:
                    self.scientist.receive_system_message(
                        "Query budget exhausted. Submit your final JSON answer now."
                    )
                    self._turns.append(self._turn_from_action(action, turn_idx, limit_event="query_budget_exhausted"))
                    continue
                result = self.world_model.process_query(content)
                self.scientist.receive_result(result)
                data_summary = ""
                if getattr(self.scientist, "_query_history", None):
                    data_summary = self.scientist._query_history[-1].get("data_summary", "")
                turn = self._turn_from_action(action, turn_idx)
                turn.query_result = result.to_dict()
                turn.data_summary = data_summary
                self._turns.append(turn)
                self._save_log(partial=True)
                if result.success:
                    self.scientist.receive_system_message(self._usage_message())
                else:
                    self.scientist.receive_system_message(
                        "Your last query failed and returned no data. Use exact JSON, exact "
                        f"measurement names, and valid intervention values. Error: {result.error_message}"
                    )
                continue

            if action_type == "answer":
                answer = content
                score = self.world_model.score_answer(content)
                turn = self._turn_from_action(action, turn_idx)
                turn.answer_score = score
                self._turns.append(turn)
                logger.info("RPG answer accepted=%s score=%s", score.get("accepted"), score)
                break

            answer = "GIVE_UP: " + content
            score = {"success": False, "accepted": False, "give_up": True, "reason": content}
            self._turns.append(self._turn_from_action(action, turn_idx))
            logger.warning("RPG scientist gave up: %s", content)
            break
        else:
            answer = "GIVE_UP: max_turns_exhausted"
            score = {"success": False, "accepted": False, "give_up": True, "reason": "max_turns_exhausted"}
            self._turns.append(RPGTurn(
                turn_index=self.max_turns,
                action_type="give_up",
                action_content="max_turns_exhausted",
                limit_event="max_turns_exhausted",
            ))

        log_path = self._save_log(partial=False, answer=answer, score=score)
        return RPGExperimentResult(
            question=self.question,
            scientist_answer=answer,
            score=score,
            turns=list(self._turns),
            total_queries=self._successful_queries_used(),
            max_queries=self.max_queries,
            resource_usage=self.world_model.get_sample_usage(),
            log_path=log_path,
        )

    def _turn_from_action(self, action: Dict[str, Any], turn_idx: int, *, limit_event: Optional[str] = None) -> RPGTurn:
        return RPGTurn(
            turn_index=turn_idx,
            action_type=action.get("type", ""),
            action_content=action.get("content", ""),
            raw_llm_response=action.get("raw_response", ""),
            reasoning=action.get("reasoning", ""),
            scientist_memory=action.get("scientist_memory", ""),
            limit_event=limit_event,
        )

    def _successful_queries_used(self) -> int:
        usage = self.world_model.get_sample_usage()
        return int(usage.get("successful_queries", 0))

    def _budget_message(self) -> str:
        info = self.world_model.get_world_info()
        budget = info.experiment_budget
        return (
            "Static RPG budget: "
            f"max_queries={self.max_queries}; "
            f"max_total_samples={budget.get('max_total_samples')} cells; "
            f"max_samples_per_query={budget.get('max_samples_per_query')} cells; "
            f"max_units_per_query={budget.get('max_units_per_query')}; "
            f"max_measurements_per_query={budget.get('max_measurements_per_query')}."
        )

    def _usage_message(self) -> str:
        usage = self.world_model.get_sample_usage()
        return (
            "Sample usage after last query: "
            f"{usage.get('sample_units_used')} {usage.get('sample_accounting')} used; "
            f"{usage.get('sample_units_remaining')} remaining; "
            f"{usage.get('successful_queries')}/{usage.get('max_queries')} queries used."
        )

    def _save_log(
        self,
        *,
        partial: bool,
        answer: str = "",
        score: Optional[Dict[str, Any]] = None,
    ) -> str:
        path = Path(self.log_dir) / f"experiment_rpg_{self._experiment_id}.json"
        payload = {
            "experiment_id": self._experiment_id,
            "partial": partial,
            "dataset_file": self.dataset_file,
            "scientist_model": self.scientist_model,
            "question": {
                "question_type": self.question.question_type,
                "question_text": self.question.question_text,
                "ground_truth": self.question.ground_truth,
                "answer_schema": self.question.answer_schema,
                "metadata": self.question.metadata,
            },
            "scientist_answer": answer,
            "score": score or {},
            "total_queries": self._successful_queries_used(),
            "max_queries": self.max_queries,
            "resource_usage": self.world_model.get_sample_usage(),
            "turns": [t.to_dict() for t in self._turns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(path)
