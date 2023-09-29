# Install vllm

```bash
!pip install vllm
```

Run in terminal, for example we will use Llama 2 7b, you can omit the host arguement however, it might cause error when working with remote server (it did in my case)

You can check vllm.entrypoints in https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py

```bash
python -m vllm.entrypoints.api_server --host 127.0.0.1 --model NousResearch/Llama-2-7b-chat-hf
```