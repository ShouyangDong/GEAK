from typing import List, Dict, Optional, Any
from textwrap import dedent


class PromptSampler:
    def __init__(self, program_db):
        """
        program_db: instance of ProgramDatabase
        """
        self.program_db = program_db

    def build_prompts(
        self,
        current_program: str,
        category: Optional[str] = None,
        few_shot_k: int = 3,
        few_shot_strategy: str = "elite",
        additional_context: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """
        Build a single few-shot prompt string for the LLM.
        - current_program: the new kernel the user supplied
        - category: operator kind to retrieve matching examples
        - few_shot_k: how many examples to include
        - few_shot_strategy: 'elite' | 'diverse' | 'similar'
        """
        instruction = (
            instruction
            or "Optimize the following Triton kernel for the target MLU. Keep function signatures identical. Use examples to guide optimizations."
        )
        examples = self.program_db.get_few_shot_examples(
            category, k=few_shot_k, strategy=few_shot_strategy
        )
        # if not enough examples for requested strategy and we have the current_program, try similarity fallback
        if len(examples) < few_shot_k:
            sim = self.program_db.find_similar_by_code(
                current_program, k=few_shot_k, category=category
            )
            # merge unique
            seen = {e["id"] for e in examples}
            for s in sim:
                if s["id"] not in seen:
                    examples.append(s)
                    seen.add(s["id"])
                if len(examples) >= few_shot_k:
                    break

        prompt_parts: List[str] = []
        prompt_parts.append("### Task\n" + instruction)
        if additional_context:
            prompt_parts.append("\n### Context\n" + additional_context)

        # Attach few-shot examples
        if examples:
            prompt_parts.append("\n### Few-shot optimized kernels (examples):\n")
            for i, ex in enumerate(examples):
                m = ex.get("metrics", {})
                meta_lines = [f"- {k}: {v}" for k, v in m.items()]
                prompt_parts.append(
                    f"--- Example {i+1} (category={ex.get('category')}):\n"
                )
                prompt_parts.append("Metrics:\n" + "\n".join(meta_lines) + "\n")
                prompt_parts.append(
                    "Code:\n```python\n" + ex.get("program", "").strip() + "\n```\n"
                )

        # Current program to optimize
        prompt_parts.append("\n### Current Program (to optimize):\n")
        prompt_parts.append("```python\n" + current_program.strip() + "\n```\n")

        # Provide constraints and instructions for autotune keys (important)
        prompt_parts.append(
            dedent(
                """
                ### Constraints and Requirements:
                - Do not change function names or parameter lists.
                - Provide a single optimized kernel variant as output.
                - Suggest autotune-friendly config keys (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) but do NOT add them as kernel parameters.
                - Keep code runnable as a Triton kernel wrapper.
                """
            )
        )

        # join and return
        return "\n".join(prompt_parts)
