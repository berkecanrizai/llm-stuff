# Using guidance library with vLLM

Define tokenizer,

```bash
from transformers import AutoTokenizer

model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
```

```bash
class vLLMOpenAI(guidance.llms.OpenAI):
    llm_name: str = "openai"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60,
                 api_key=None, api_type="open_ai", api_base=None, api_version=None, deployment_id=None,
                 temperature=0.0, chat_mode="auto", organization=None, rest_call=False,
                 allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>"},
                 token=None, endpoint=None, encoding_name=None, tokenizer=None):

        # map old param values
        # TODO: add deprecated warnings after some time
        if token is not None:    
            if api_key is None:
                api_key = token
        if endpoint is not None:
            if api_base is None:
                api_base = endpoint

        # fill in default model value
        if model is None:
            model = os.environ.get("OPENAI_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.openai_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # fill in default deployment_id value
        if deployment_id is None:
            deployment_id = os.environ.get("OPENAI_DEPLOYMENT_ID", None)

        # auto detect chat completion mode
        chat_mode = False
        
        # fill in default API key value
        if api_key is None: # get from environment variable
            api_key = os.environ.get("OPENAI_API_KEY", getattr(openai, "api_key", None))
        if api_key is not None and not api_key.startswith("sk-") and os.path.exists(os.path.expanduser(api_key)): # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None: # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass
        if organization is None:
            organization = os.environ.get("OPENAI_ORGANIZATION", None)
        # fill in default endpoint value
        if api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", None) or os.environ.get("OPENAI_ENDPOINT", None) # ENDPOINT is deprecated

        self._tokenizer = tokenizer
        self.chat_mode = chat_mode
        
        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.deployment_id = deployment_id
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.organization = organization
        self.rest_call = rest_call
        self.endpoint = endpoint

        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }
```

# initialize the custom model for inference, we change base url and key.

```bash
llm = vLLMOpenAI(
    "NousResearch/Llama-2-7b-chat-hf",
    api_key='EMPTY',
    api_base="http://localhost:8000/v1",
    tokenizer=tokenizer,
    chat_mode=False,
    rest_call=False,
)
```

Set the default llm,

```bash
guidance.llm = llm
```

Run,
```bash
program = guidance("""My favorite flavor is{{gen 'flavor' max_tokens=50 stop="." save_stop_text=True}}""", caching=False)
out = program()
# My favorite flavor is chocolate, but I also enjoy other flavors like strawberry and vanilla

print(out.get('flavor'))
# ' chocolate, but I also enjoy other flavors like strawberry and vanilla'
```