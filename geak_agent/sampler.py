class PromptSampler:
    def __init__(self, program_db):
        self.program_db = program_db

    def build_prompts(self, parents):
        prompts = []
        for parent in parents:
            program = parent["program"]
            metrics = parent["metrics"]
            prompt = f"""
### Parent Program
{program}

### Performance Metrics
- Speedup: {metrics['speedup']}
- Efficiency: {metrics['efficiency']}

### Task
Optimize the above program for better performance while maintaining correctness.
"""
            prompts.append(prompt)
        return prompts
