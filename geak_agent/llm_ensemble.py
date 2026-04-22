from geak_agent.models.OpenAI import StandardOpenAIModel

class LLMEnsemble:
    def __init__(self):
        self.models = {
            "gpt-4": StandardOpenAIModel(),
        }

    def generate_variants(self, prompts):
        variants = []
        for prompt in prompts:
            for model_name, model in self.models.items():
                try:
                    code = model.generate(prompt)
                    variants.append({"program": code, "model": model_name})
                except Exception as e:
                    print(f"Error generating with {model_name}: {e}")
        return variants
