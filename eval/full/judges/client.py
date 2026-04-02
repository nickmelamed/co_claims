import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMClient:
    def __init__(self, model_name: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map={"": self.device}
        )

    def chat(self, messages, temperature=0.0, max_tokens=512):
        prompt = self._format_chat(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return only the generated part (optional cleanup)
        return decoded[len(prompt):].strip()

    def _format_chat(self, messages):
        # Simple chat format (works across most instruct models)
        formatted = ""
        for m in messages:
            role = m["role"]
            content = m["content"]

            if role == "system":
                formatted += f"[SYSTEM]\n{content}\n"
            elif role == "user":
                formatted += f"[USER]\n{content}\n"
            elif role == "assistant":
                formatted += f"[ASSISTANT]\n{content}\n"

        formatted += "[ASSISTANT]\n"
        return formatted