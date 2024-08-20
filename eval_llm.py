from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import utils
from deepeval.benchmarks import MMLU
from typing import List
from deepeval.benchmarks.tasks import MMLUTask
from peft import PeftModelForCausalLM

class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0][-1]
        # generated_ids = model(**model_inputs).logits[0, -1, :].argmax()
        # return self.tokenizer.decode(generated_ids)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, promtps: List[str]) -> List[str]:
        model = self.load_model()
        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer(promtps, return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return "Mistral 7B"

utils.set_proxy()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="cache")
model = PeftModelForCausalLM.from_pretrained(model=model, model_id="/home/fuwenjie/Extraction-LLMs/defend_llms/meta-llama/Llama-2-7b-hf/wjfu99/WikiMIA-24/128/checkpoint-250", cache_dir="cache")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="cache")

tokenizer.pad_token_id = tokenizer.eos_token_id

mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
# print(mistral_7b("Write me a joke"))
benchmark = MMLU(
    # tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE],
    n_shots=5
)
# When you set batch_size, outputs for benchmarks will be generated in batches
# if `batch_generate()` is implemented for your custom LLM
results = benchmark.evaluate(model=mistral_7b)
print("Overall Score: ", results)