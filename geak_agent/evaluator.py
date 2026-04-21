from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict


class BaseEvaluator:
    def evaluate(self, program: Dict[str, Any]) -> bool:
        """Fallback evaluator: accept everything but log for visibility."""
        return True


class CorrectnessEvaluator(BaseEvaluator):
    def evaluate(self, program):
        # TODO: replace with real correctness check
        return True


class PerformanceEvaluator(BaseEvaluator):
    def evaluate(self, program):
        # TODO: replace with real performance benchmark
        return True


class EvaluatorPool:
    def __init__(self):
        self.evaluators = [CorrectnessEvaluator(), PerformanceEvaluator()]

    def evaluate(self, programs):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_program, program) for program in programs
            ]
            for future in futures:
                results.append(future.result())
        return results

    def _evaluate_program(self, program):
        for evaluator in self.evaluators:
            try:
                ok = evaluator.evaluate(program)
            except Exception:
                return None
            if not ok:
                return None  # Filter out invalid programs
        return program
