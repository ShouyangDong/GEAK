from geak_agent.database import ProgramDatabase
from geak_agent.sampler import PromptSampler
from geak_agent.evaluator import EvaluatorPool
from geak_agent.llm_ensemble import LLMEnsemble


class Orchestrator:
    def __init__(self, dataset, max_iterations=10, islands=3):
        self.dataset = dataset
        self.max_iterations = max_iterations
        self.islands = islands
        self.program_db = ProgramDatabase()
        self.prompt_sampler = PromptSampler(self.program_db)
        self.llm_ensemble = LLMEnsemble()
        self.evaluator_pool = EvaluatorPool()

    def run(self):
        for iteration in range(self.max_iterations):
            print(f"=== Iteration {iteration + 1}/{self.max_iterations} ===")
            for island in range(self.islands):
                print(f"--- Island {island + 1}/{self.islands} ---")
                self.optimize_island(island)

    def optimize_island(self, island_id):
        # 1. Selection: Get elite + diverse parents
        parents = self.program_db.select_parents(elite_count=2, diverse_count=3)

        # 2. Mutation: Generate new variants using LLM
        prompts = self.prompt_sampler.build_prompts(parents)
        new_variants = self.llm_ensemble.generate_variants(prompts)

        # 3. Evaluation: Validate correctness and benchmark performance
        evaluated_variants = self.evaluator_pool.evaluate(new_variants)

        # 4. Update: Refresh the program database
        self.program_db.update(evaluated_variants)
