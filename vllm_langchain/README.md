

```bash
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoModelForCausalLM, GenerationConfig
```

load the generation config (Huggingface)

```bash
generation_config = GenerationConfig.from_pretrained("../llama2_safe")

default_config['max_tokens'] = default_config['max_length']
```

will use for filtering args from config, otherwise vLLM raises error.

from vllm import SamplingParams

sp = SamplingParams()

```bash
import requests
import json

def request_get(prompt, generation_args={}, **kwargs):
    url = 'http://localhost:8000/generate'

    generation_args = {k: v for k, v in generation_args.items() if k in sp.__dict__}

    data = {
        "prompt": prompt,
        "use_beam_search": False,
        "n": 1,
    } | generation_args | kwargs
    
    data_json = json.dumps(data)
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=data_json, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
```



```bash
request_get('my name is', generation_args=default_config)

#produces: {'text': ['my name is john and i am a 35 year old man from the united states. i have']}
```

```bash
# define custom LLM for langchain
class CustomLLM(LLM):
    model_name: str='custom'
    max_len: int=8000
    config: dict=default_config

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        #print(self.config)
        
        return request_get(prompt[: self.max_len], generation_args=self.config, **kwargs)['text'][0]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
```
Example use,
```bash
my_llm = CustomLLM()

my_llm('I usually code in', temperature=1)
# "I usually code in Java, but I'm interested in learning Go. Here are some resources I've found to"

my_llm('I usually code in', temperature=0.001)
# "I usually code in Python, but I'm interested in learning more about Rust and its ecosystem. Here"

my_llm('I usually code in', temperature=0.001, max_tokens=100)
#"I usually code in Python, but I'm interested in learning more about Rust and its ecosystem. Here are some resources I've found helpful:\n\n1. The Rust Programming Language: This is the official book on Rust, written by the language's creators. It covers the language's syntax, standard library, and best practices.\n2. Rust by Example: This book provides a gentle introduction to Rust, with a focus on practical examples and exercis"
```


# Bonus
Callbacks and Agents

```bash
from langchain import PromptTemplate, LLMChain

pr = """<s>[INST] <<SYS>>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

<</SYS>>
Respond to user. User: {input} [/INST]"""

llm_chain = LLMChain(llm=my_llm, prompt=PromptTemplate.from_template(pr))

response = llm_chain.run(input='How are you?')

#"<s>[INST] <<SYS>>\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n<</SYS>>\nRespond to user. User: How are you? [/INST]  Hello! I'm doing well, thank you for asking! How about you?"
```