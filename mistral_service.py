from pydub import AudioSegment
import os
import whisper
from tqdm import tqdm
import math
import string
import random
import requests as r
import asyncio
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from pymystem3 import Mystem
from nltk.corpus import stopwords
import json
from string import punctuation
model = whisper.load_model("medium")

llm_origin = 'https://51cc-178-154-246-234.ngrok-free.app'
#model = whisper.load_model("large")
model = None
origin = 'http://192.168.62.4:8000/api/message/'
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


def make_llm_rec(query: str):
    payload = json.dumps({
        "data": query
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = r.request("POST", llm_origin, headers=headers, data=payload)
    return response.text

def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    return text


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    return text

def make_request(uuid, data):
    print(data)
    data = r.post(origin + uuid + '/', data)

def generate_slug():
    return ''.join([random.choice(string.ascii_lowercase) for _ in range(6)])


def split_audio(input_file, output_folder, duration):
    audio = AudioSegment.from_mp3(input_file)
    total_length = len(audio)
    num_parts = math.ceil(total_length / (duration * 1000))
    lst_i = 0

    for i in range(num_parts):
        start = i * duration * 1000
        end = (i + 1) * duration * 1000
        split_audio = audio[start:end]
        output_path = os.path.join(output_folder, f"part_{i+1}.mp3")
        split_audio.export(output_path, format="mp3")
        lst_i = i
    return lst_i


def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    return result.text


def make_search(sentences, termins, uuid):
    print(sentences)
    store = InMemoryDocumentStore(use_bm25=True)
    store.write_documents([
        {
            'content': preprocess_text(sentence['content']),
            'meta': sentence
        } for sentence in sentences
    ])
    retriever = BM25Retriever(store, top_k=1)
    for term in termins:
        meta = retriever.retrieve(preprocess_text(term))[0].meta
        make_request(uuid, {'data': json.dumps({
            'time_from': meta['from'],
            'time_to': meta['to'],
            'entry_sentence': meta['content'],
            'uuid': uuid,
            'name': term
        }, ensure_ascii=False), 'type': 'time'})


async def make_summary(text, uuid):
    text = ' '.join(text.split()[0:1000])
    res = make_llm_rec(f"{text}\nЭто отрывок из урока. Напиши по нему краткий конспект")
    make_request(uuid, {'data': res, 'type': 'summary'})
    
    res = make_llm_rec(f'{text}\nЭто отрывок из урока. Предположи название урока. Напиши название, ничего больше писать не нужно')
    make_request(uuid, {'data': res, 'type': 'name'})

    



async def make_llm_query(uuid, query, sentences):
    text = make_llm_rec(f"{query}\nЭто отрывок из урока. выдели из него все определения. Напиши их ненумерованным списком. Ничего кроме нумерованного списка возвращять не нужно. Всегда отдавай значение и определение через двоеточие")
    
    for item in text.split('\n'):
        make_request(uuid, {"type": "terms", 'data': item})
    make_search(sentences, text.split('\n'), uuid)
    return text

async def run(filepath, uuid):
    global model
    model = whisper.load_model("medium")
    loop = asyncio.get_event_loop()

    delta = 5

    name = filepath.split('\\')[-1].split('.')[0]
    lst = split_audio(filepath, f'./chunks/{name}', delta)
    
    res = []
    local_res = []

    sentences = []
    sentence = {'content': '', 'from': 0, 'to': 0}
    
    for i in tqdm(range(1, lst+1)):
        text = inference(f'./chunks/{name}/part_{i}.mp3')
        res.append(text)
        make_request(uuid, {
            'type': 'trans',
            'data': text
        })
        print(text)
        if '.' in text:
            try:
                text_left, text_right = text.split('.')[0:2]
                sentence['content'] += (text_left)
                sentence['to'] = i * delta
                sentences.append(sentence)
                sentence = {'content': text_right, 'from': i * delta, 'to': i * delta}
            except Exception as e: 
                print(e)
        local_res.append(text)
        if len(' '.join(local_res).split(' ')) >= 100:
            loop.create_task(make_llm_query(uuid, local_res, sentences))
            local_res = []
            sentences  = []
            
    if len(local_res):
        loop.create_task(make_llm_query(uuid, local_res, sentences))
        pass
    make_request(uuid, {'type': 'trans_end', 'data': 'true'})
    loop.create_task(make_summary(' '.join(res), uuid))