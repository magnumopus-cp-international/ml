from llama_cpp import Llama
from fastapi import FastAPI
from pydantic import BaseModel

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13
top_k=30
top_p=0.9
temperature=0.2
repeat_penalty=1.1

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

app = FastAPI()



def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


model = Llama(
    model_path='./model-q4_K.gguf.1',
    n_ctx=4096,
    n_parts=1,
)

system_tokens = get_system_tokens(model)
tokens = system_tokens
model.eval(tokens)

def make_message(text):
    global tokens, model
    
    user_message = text
    message_tokens = get_message_tokens(model=model, role="user", content=user_message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    full_prompt = model.detokenize(tokens)
    generator = model.generate(
                tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repeat_penalty
    )
    acc_tokens = ''
    for token in generator:

        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        acc_tokens += token_str
        tokens.append(token)
        if token == model.token_eos():
            break

    print(acc_tokens)
    tokens = system_tokens

def make_message(text):
    global tokens, model
    
    user_message = text
    message_tokens = get_message_tokens(model=model, role="user", content=user_message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    full_prompt = model.detokenize(tokens)
    generator = model.generate(
                tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repeat_penalty
    )
    acc_tokens = ''
    for token in generator:

        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        acc_tokens += token_str
        tokens.append(token)
        if token == model.token_eos():
            break
    tokens = system_tokens
    return acc_tokens


class LLmQueryModel(BaseModel):
    data: str

@app.post("/")
def make_llm_query(data: LLmQueryModel):
    return make_message(data.data)