class BaseJudge:
    def evaluate(self, prompt: str) -> dict:
        raise NotImplementedError