from .tool_class_base import ToolClass
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen(ToolClass):
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )

    def get_res(self, messages, functions, extra_args={}, more_info=None):
        text = self.tokenizer.apply_chat_template(messages, tools=functions, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.decode_res(inputs, outputs)
    
    def _get_res(self, messages):
        # outputs = self.pipeline(messages, max_new_tokens=512)
        print("just messages")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        return model_inputs, outputs

    def decode_res(self, prompt, outputs):
        # print(len(prompt))
        # print(type(outputs), outputs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt.input_ids, outputs)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
