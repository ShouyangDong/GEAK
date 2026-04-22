from geak_agent.controller import Orchestrator
from geak_agent.database import ProgramDatabase
from geak_agent.sampler import PromptSampler
from geak_agent.llm_ensemble import LLMEnsemble
from geak_agent.evaluator import EvaluatorPool


# Mock classes for testing
class MockCorrectnessEvaluator:
    def evaluate(self, program):
        # Simulate correctness validation
        return "correct" in program


class MockPerformanceEvaluator:
    def evaluate(self, program):
        # Simulate performance benchmarking
        return "optimized" in program


class MockLLMModel:
    def generate(self, prompt):
        # Simulate LLM generating a new kernel variant
        return f"{prompt}\n# Optimized kernel code"


# Replace real evaluators and LLMs with mocks for testing
class MockEvaluatorPool(EvaluatorPool):
    def __init__(self):
        self.evaluators = [MockCorrectnessEvaluator(), MockPerformanceEvaluator()]


class MockLLMEnsemble(LLMEnsemble):
    def __init__(self):
        self.models = {"mock-llm": MockLLMModel()}


# Test dataset
mock_dataset = [
    {"program": "kernel_1_code", "metrics": {"speedup": 1.0, "efficiency": 0.8}},
    {"program": "kernel_2_code", "metrics": {"speedup": 1.2, "efficiency": 0.85}},
]


# Test the workflow
def test_workflow():
    print("=== Testing Workflow ===")

    # Initialize components
    program_db = ProgramDatabase()
    for data in mock_dataset:
        program_db.add_program(data["program"], data["metrics"], category=data.get("category", "unknown"))

    prompt_sampler = PromptSampler(program_db)
    llm_ensemble = MockLLMEnsemble()
    evaluator_pool = MockEvaluatorPool()

    # Initialize orchestrator
    orchestrator = Orchestrator(
        dataset=mock_dataset,
        max_iterations=2,
        islands=1,
    )
    orchestrator.program_db = program_db
    orchestrator.prompt_sampler = prompt_sampler
    orchestrator.llm_ensemble = llm_ensemble
    orchestrator.evaluator_pool = evaluator_pool

    # Run the workflow
    orchestrator.run()

    # Check results
    print("\n=== Final Program Database ===")
    for program in orchestrator.program_db.programs:
        print(program)


if __name__ == "__main__":
    test_workflow()
