from typing import Optional, Annotated

from fastapi import FastAPI, File, UploadFile, Form
from mistral_service import run, generate_slug, make_llm_query, make_summary
import asyncio
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload_audio")
async def read_item(uuid=Form(), file=File()):
    print(uuid)
    if file:
        slug = generate_slug()
        print(slug)
        with open(f'./files/{slug}.mp3', 'wb') as f:
            content = await file.read()
            f.write(content)
            filepath = f'./files/{slug}.mp3'
    loop = asyncio.get_event_loop()
    loop.create_task(run(filepath, uuid))
    return {}


class SummaryItem(BaseModel):
    query: str
    uuid: str


@app.post('/summary')
async def llm_query(data: SummaryItem):
    loop = asyncio.get_event_loop()
    splitted_query = data.query.split()
    for i in range(0, len(splitted_query), 1000):
        loop.create_task(make_llm_query(data.uuid, ' '.join(splitted_query[i:min(i+1000, len(splitted_query))]), 
                                            [{'content': data.query.split('.')[i], 'from': i*5-5, 'to': i*5} for i in range(len(data.query.split('.')))]
                                        ))
    loop.create_task(make_summary(data.query, data.uuid))