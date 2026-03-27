"""
orchestrator.py

Manages the interaction loop between the Scientist Agent and the World Model.

Responsibilities:
    - Present the initial problem (variables + question) to the Scientist
    - Relay queries from Scientist to World Model
    - Enforce query budget
    - Log all interactions
    - Collect and evaluate the final answer

Usage:
    orchestrator = Orchestrator(
        world_model=world,
        scientist=scientist,
        question=question,
        max_queries=10,
    )
    result = orchestrator.run()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from schemas import Question, QueryResult, WorldInfo

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Record of a single turn in the interaction."""
    turn_number: int
    turn_type: str  # "query", "answer", "system"
    scientist_input: str
    world_output: Optional[str]
    parsed_query: Optional[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Enhanced logging fields
    raw_llm_response: Optional[str] = None
    reasoning: Optional[str] = None
    scientist_memory: Optional[str] = None
    data_summary: Optional[str] = None
    # Full multi-turn conversation (populated by CoderScientistAgent)
    llm_transcript: Optional[List[Dict[str, Any]]] = None
    # Structured record of each code execution: [{"code": ..., "output": ...}, ...]
    code_rounds: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "turn_number": self.turn_number,
            "turn_type": self.turn_type,
            "scientist_input": self.scientist_input,
            "world_output": self.world_output,
            "parsed_query": self.parsed_query,
            "timestamp": self.timestamp,
        }
        if self.raw_llm_response is not None:
            d["raw_llm_response"] = self.raw_llm_response
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.scientist_memory is not None:
            d["scientist_memory"] = self.scientist_memory
        if self.data_summary is not None:
            d["data_summary"] = self.data_summary
        if self.llm_transcript is not None:
            d["llm_transcript"] = self.llm_transcript
        if self.code_rounds is not None:
            d["code_rounds"] = self.code_rounds
        return d


@dataclass
class ExperimentResult:
    """Final result of an experiment run."""
    question: Question
    scientist_answer: str
    ground_truth: Any
    is_correct: bool
    turns: List[Turn]
    total_queries: int
    max_queries: int
    final_scientist_memory: str = ""
    # Provenance — set from Orchestrator fields
    scientist_model: Optional[str] = None
    world_model_name: Optional[str] = None
    dataset_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        # Provenance at the top so it's immediately visible in the file
        if self.scientist_model is not None:
            d["scientist_model"] = self.scientist_model
        if self.world_model_name is not None:
            d["world_model"] = self.world_model_name
        if self.dataset_file is not None:
            d["dataset_file"] = self.dataset_file
        d.update({
            "question": self.question.to_dict(),
            "scientist_answer": self.scientist_answer,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "total_queries": self.total_queries,
            "max_queries": self.max_queries,
            "final_scientist_memory": self.final_scientist_memory,
            "turns": [t.to_dict() for t in self.turns],
        })
        return d
    
    def save(self, filepath: str) -> None:
        """Save result to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class Orchestrator:
    """
    Manages the interaction between Scientist Agent and World Model.
    
    The orchestrator:
        1. Presents the problem to the scientist
        2. Processes scientist's queries through the world model
        3. Tracks budget and history
        4. Collects the final answer
    """
    world_model: Any  # WorldModel instance
    scientist: Any    # ScientistAgent instance
    question: Question
    max_queries: int = 10
    log_dir: str = "./experiment_logs"
    # Provenance — logged into the experiment JSON for reproducibility
    scientist_model: Optional[str] = None
    world_model_name: Optional[str] = None
    dataset_file: Optional[str] = None

    # Internal state
    _turns: List[Turn] = field(default_factory=list, init=False)
    _query_count: int = field(default=0, init=False)
    _experiment_id: str = field(default="", init=False)
    
    def __post_init__(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run(self) -> ExperimentResult:
        """
        Run the full experiment loop.
        
        Returns:
            ExperimentResult with the outcome
        """
        logger.info(f"Starting experiment {self._experiment_id}")
        logger.info(f"Question: {self.question.question_text}")
        logger.info(f"Budget: {self.max_queries} queries")
        
        # Step 1: Initialize scientist with world info and question
        world_info = self.world_model.get_world_info()
        self.scientist.initialize(world_info, self.question, self.max_queries)
        
        # Step 2: Main interaction loop
        while True:
            # Get scientist's next action
            action = self.scientist.get_next_action()

            if action["type"] == "query":
                # Process query
                if self._query_count >= self.max_queries:
                    # Budget exhausted - force answer
                    logger.warning("Query budget exhausted, forcing answer")
                    self._notify_budget_exhausted()
                    continue

                result = self._process_query(action)
                self.scientist.receive_result(result)

                # Update the turn's data_summary with the full statistical summary
                # computed by the scientist during receive_result
                if self.scientist._query_history:
                    latest = self.scientist._query_history[-1]
                    self._turns[-1].data_summary = latest.get("data_summary")

            elif action["type"] == "answer":
                # Log the final answer turn with reasoning/memory
                self._log_answer_turn(action)
                answer = action["content"]
                logger.info(f"Scientist answer: {answer}")
                break

            elif action["type"] == "give_up":
                self._log_answer_turn(action)
                answer = "GIVE_UP: " + action.get("content", "No reason given")
                logger.warning(f"Scientist gave up: {answer}")
                break

            else:
                logger.error(f"Unknown action type: {action['type']}")
                continue

        # Step 3: Evaluate answer
        is_correct = self._evaluate_answer(answer, self.question.ground_truth)

        # Get final scientist memory
        final_memory = getattr(self.scientist, '_scientist_memory', '')

        result = ExperimentResult(
            question=self.question,
            scientist_answer=answer,
            ground_truth=self.question.ground_truth,
            is_correct=is_correct,
            turns=self._turns,
            total_queries=self._query_count,
            max_queries=self.max_queries,
            final_scientist_memory=final_memory,
            scientist_model=self.scientist_model,
            world_model_name=self.world_model_name,
            dataset_file=self.dataset_file,
        )
        
        # Save logs
        log_path = Path(self.log_dir) / f"experiment_{self._experiment_id}.json"
        result.save(str(log_path))
        logger.info(f"Experiment complete. Correct: {is_correct}. Log: {log_path}")
        
        return result
    
    def _process_query(self, action: Dict[str, Any]) -> QueryResult:
        """Process a query through the world model, logging enhanced metadata."""
        query = action["content"]

        result = self.world_model.process_query(query)

        # Only consume budget on successful queries so that parse errors,
        # validation errors, etc. don't penalise the scientist.
        if result.success:
            self._query_count += 1

        logger.info(f"Query {self._query_count}/{self.max_queries} (success={result.success}): {query[:100]}...")

        # Compute data summary string for logging (scientist will also compute it)
        data_summary = None
        if result.success and result.data_file:
            # The scientist's receive_result will compute the full summary;
            # we grab the scientist's most recent history entry after receive_result
            # is called. For now, store what we can from the result.
            data_summary = result.preview

        # Log the turn with enhanced metadata
        turn = Turn(
            turn_number=len(self._turns) + 1,
            turn_type="query",
            scientist_input=query,
            world_output=result.to_xml(),
            parsed_query=result.query.to_dict() if result.success else None,
            raw_llm_response=action.get("raw_response"),
            reasoning=action.get("reasoning"),
            scientist_memory=action.get("scientist_memory"),
            data_summary=data_summary,
            llm_transcript=action.get("llm_transcript"),
            code_rounds=action.get("code_rounds"),
        )
        self._turns.append(turn)

        return result

    def _log_answer_turn(self, action: Dict[str, Any]) -> None:
        """Log an answer or give_up turn with enhanced metadata."""
        turn = Turn(
            turn_number=len(self._turns) + 1,
            turn_type=action["type"],
            scientist_input=action["content"],
            world_output=None,
            parsed_query=None,
            raw_llm_response=action.get("raw_response"),
            reasoning=action.get("reasoning"),
            scientist_memory=action.get("scientist_memory"),
            llm_transcript=action.get("llm_transcript"),
            code_rounds=action.get("code_rounds"),
        )
        self._turns.append(turn)
    
    def _notify_budget_exhausted(self) -> None:
        """Notify scientist that query budget is exhausted."""
        message = f"""<budget_exhausted>
You have used all {self.max_queries} queries. 
You must now provide your final answer based on the data you have collected.
</budget_exhausted>"""
        self.scientist.receive_system_message(message)
    
    def _evaluate_answer(self, answer: str, ground_truth: Any) -> bool:
        """
        Evaluate if the scientist's answer matches ground truth.
        
        This is a simple implementation - can be extended for different question types.
        """
        # Normalize answer
        answer_normalized = answer.strip().lower()
        
        # Handle different ground truth types
        if isinstance(ground_truth, bool):
            # Yes/No questions
            positive = ["yes", "true", "1", "correct"]
            negative = ["no", "false", "0", "incorrect"]
            
            answer_is_positive = any(p in answer_normalized for p in positive)
            answer_is_negative = any(n in answer_normalized for n in negative)
            
            if answer_is_positive and not answer_is_negative:
                return ground_truth == True
            elif answer_is_negative and not answer_is_positive:
                return ground_truth == False
            else:
                # Ambiguous
                return False
                
        elif isinstance(ground_truth, set):
            # Set questions (e.g., "list all ancestors")
            # Try to extract variable names from answer
            import re
            # Find words that match variable names
            all_vars = set(self.world_model.simulator.get_nodes())
            found_vars = set()
            for var in all_vars:
                if var.lower() in answer_normalized:
                    found_vars.add(var)
            
            return found_vars == ground_truth
            
        elif isinstance(ground_truth, list):
            # Convert to set for comparison
            return self._evaluate_answer(answer, set(ground_truth))
            
        elif isinstance(ground_truth, str):
            # String matching
            return ground_truth.lower() in answer_normalized
            
        else:
            # Fallback: string comparison
            return str(ground_truth).lower() in answer_normalized


# -----------------------------------------------------------------------------
# Simple interaction loop for testing (without full scientist agent)
# -----------------------------------------------------------------------------

def run_interactive_session(world_model: Any, question: Question, max_queries: int = 10):
    """
    Run an interactive session where a human plays the scientist.
    
    Useful for testing the world model.
    """
    print("\n" + "="*60)
    print("CAUSAL DISCOVERY INTERACTIVE SESSION")
    print("="*60)
    
    # Show world info
    world_info = world_model.get_world_info()
    print("\n" + world_info.to_xml())
    
    # Show question
    print("\n" + question.to_xml())
    
    print(f"\nYou have {max_queries} queries. Type 'answer' to submit your answer.")
    print("="*60)
    
    query_count = 0
    
    while query_count < max_queries:
        print(f"\n[Query {query_count + 1}/{max_queries}]")
        user_input = input("Your query (or 'answer'): ").strip()
        
        if user_input.lower() == "answer":
            answer = input("Your final answer: ").strip()
            print(f"\nYour answer: {answer}")
            print(f"Ground truth: {question.ground_truth}")
            break
        
        if not user_input:
            continue
        
        result = world_model.process_query(user_input)
        print("\n" + result.to_xml())
        query_count += 1
    
    else:
        print("\nQuery budget exhausted!")
        answer = input("Your final answer: ").strip()
        print(f"\nYour answer: {answer}")
        print(f"Ground truth: {question.ground_truth}")


if __name__ == "__main__":
    # Quick test with interactive mode
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("bif_path", help="Path to BIF file")
    parser.add_argument("--max-queries", "-n", type=int, default=5)
    args = parser.parse_args()
    
    from simulator import BNSimulator
    from world_model_causal import WorldModel, QwenLLM
    from schemas import Question
    
    # Load
    sim = BNSimulator.from_bif(args.bif_path)
    
    print("Loading LLM (this may take a moment)...")
    llm = QwenLLM()
    
    world = WorldModel(simulator=sim, llm=llm)
    
    # Example question
    question = Question(
        question_type="direct_edge",
        question_text="Is there a direct causal edge from 'smoke' to 'lung'?",
        ground_truth=True,
    )
    
    run_interactive_session(world, question, args.max_queries)
