# getting structured and parsable output is quite a challenge with models that are stochastic by nature however,
# during auto regression we can choose the next token to force model to behave in certain ways

# guidance library is a good one for such application

We define a regex that model follows,

```bash
# ...

import guidance

llama = guidance.llms.Transformers(model=model, tokenizer=tokenizer, max_length=350, config={'temperature': 0},)
guidance.llm = llama

user_input = 8
    
program = guidance(f"""<s>[INST] <<SYS>>
You are virtual assistant, be clear and precise.
<</SYS>>
Given a users desired number, list 5 preceding numbers as list of strings.
Don't explain response, give parsable python list only and only list dish names and nothing else.
User input: {user_input} Numbers:[/INST]""" + """{{gen 'ls' pattern="^\['.*'\]$" save_stop_text=False}}""", ) # caching=False

out = program()
output_list = ast.literal_eval(out.get('ls'))

#['3', '4', '5', ...]
```
