import asyncio
import re
import logging
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from g4f.client import Client
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client()

top_models = [
    "gpt-4",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.5",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-r1-turbo",
    "deepseek-v3-0324",
    "deepseek-v3-0324-turbo",
    "deepseek-r1-0528",
    "deepseek-r1-0528-turbo",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-32b",
    "grok-3",
    "grok-3-r1",
]

async def request_ai(prompt: str) -> Dict[str, Any]:
    for model in top_models:
        logger.info(f"Trying model '{model}'...")
        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        web_search=False,
                        stream=False
                    )
                ),
                timeout=90
            )
            elapsed = time.time() - start_time
            content = response.choices[0].message.content.strip()
            logger.info(f"Model '{model}' returned a response successfully in {elapsed:.2f} seconds.")
            return {"model": model, "content": content, "time": elapsed}

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"Model '{model}' timed out after {elapsed:.2f} seconds.")
            continue

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Model '{model}' failed after {elapsed:.2f} seconds with error: {e}")
            continue

    logger.warning("No model returned a valid response.")
    return {"model": None, "content": "Brak odpowiedzi od żadnego modelu.", "time": None}

def parse_questions(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"Start:\s*(.*?)\s*End:\s*OptionsStart:\s*(.*?)\s*(\d)\s*OptionsEnd:",
        re.DOTALL
    )
    questions = []
    matches = pattern.findall(text)

    for match in matches:
        question_text = match[0].strip()
        options_text = match[1].strip()
        correct_index = int(match[2])

        options = [opt.strip() for opt in options_text.splitlines() if opt.strip()]

        if len(options) != 4:
            continue

        questions.append({
            "question": question_text,
            "options": options,
            "correctOptionIndex": correct_index
        })

    return questions

prompt_question = """
Stwórz dokładnie 10 pytań z geografii dla szkoły średniej, temat: "Rolnictwo, leśnictwo, rybactwo".  
Każde pytanie powinno losowo pochodzić z jednej z poniższych podtematów:
1. Czynniki rozwoju rolnictwa
2. Główne obszary upraw
3. Chów zwierząt
4. Lasy na Ziemi. Gospodarka leśna
5. Rybactwo

Odpowiedzi muszą być **dokładnie** w poniższym formacie:

Start:
[Tutaj wstaw pytanie]
End:
OptionsStart:
Option1
Option2
Option3
Option4
correctOptionIndex
OptionsEnd:

Instrukcje:
1. Wygeneruj dokładnie 10 oddzielnych pytań w powyższym formacie, jedno po drugim, bez dodatkowego tekstu między nimi.  
2. Każde pytanie musi mieć dokładnie 4 realistyczne, unikalne i sensowne odpowiedzi w kolejności Option1–Option4.  
3. correctOptionIndex to numer poprawnej odpowiedzi od 0 do 3.  
4. Nie dodawaj żadnych komentarzy ani wprowadzeń – tylko 10 pytań w dokładnym formacie Start/End oraz OptionsStart/OptionsEnd.  
5. Pytania muszą być powiązane z tematami podanymi powyżej, losowo wybierając temat dla każdego pytania.
"""

prompt_true_false_binary = """
Stwórz dokładnie 10 pytań typu Prawda/Fałsz z geografii dla szkoły średniej.  
Każde pytanie powinno losowo pochodzić z jednej z poniższych podtematów:
1. Czynniki rozwoju rolnictwa
2. Główne obszary upraw
3. Chów zwierząt
4. Lasy na Ziemi. Gospodarka leśna
5. Rybactwo

Odpowiedzi muszą być **dokładnie** w poniższym formacie i oznaczone 1 (prawda) lub 0 (fałsz):

Start:
[Tutaj pytanie Prawda/Fałsz]
End:
answerStart:
1
answerEnd:

Instrukcje:
1. Wygeneruj dokładnie 10 oddzielnych pytań w powyższym formacie, jedno po drugim.  
2. Nie dodawaj komentarzy ani wprowadzeń – tylko 10 pytań w dokładnym formacie Start/End oraz answerStart/answerEnd.  
3. Każda odpowiedź powinna być poprawnie oceniona jako 1 (prawda) lub 0 (fałsz).  
4. Pytania muszą być powiązane z tematami podanymi powyżej, losowo wybierając temat dla każdego pytania.
"""

prompt_open = """
Stwórz dokładnie 5 pytań otwartych z geografii dla szkoły średniej.  
Każde pytanie powinno losowo pochodzić z jednej z poniższych podtematów:
1. Czynniki rozwoju rolnictwa
2. Główne obszary upraw
3. Chów zwierząt
4. Lasy na Ziemi. Gospodarka leśna
5. Rybactwo

Odpowiedzi **nie są potrzebne** – tylko pytania.  
Każde pytanie musi być w formacie:

Start:
[Tutaj pytanie]
End:

Instrukcje:
1. Wygeneruj dokładnie 5 pytań, jedno po drugim, bez dodatkowego tekstu między nimi.
2. Zachowaj dokładnie strukturę Start/End.
3. Pytania muszą być realistyczne i losowo dobrane z podanych tematów.
"""

def parse_open_questions(raw_text: str):
    questions = []
    parts = raw_text.split("Start:")
    for part in parts[1:]:
        try:
            question_text = part.split("End:")[0].strip()
            questions.append(question_text)
        except IndexError:
            continue
    return questions

def parse_true_false_binary(raw_text: str):
    questions = []
    pattern = re.compile(
        r"Start:\s*(.*?)\s*End:\s*answerStart:\s*(1|0)\s*answerEnd:",
        re.DOTALL
    )
    matches = pattern.findall(raw_text)

    for match in matches:
        question_text = match[0].strip()
        answer_value = int(match[1])
        questions.append({
            "question": question_text,
            "answer": answer_value
        })

    return questions

@app.get("/chatABCD")
async def chat():
    result = await request_ai(prompt_question)
    questions = parse_questions(result["content"])
    return JSONResponse({
        "model": result["model"],
        "questions": questions
    })


@app.get("/chatTRUEFALSE")
async def chat_true_false():
    result = await request_ai(prompt_true_false_binary)
    questions = parse_true_false_binary(result["content"])
    return JSONResponse({
        "model": result["model"],
        "questions": questions
    })

@app.get("/chatOPEN")
async def chat_open():
    result = await request_ai(prompt_open)
    questions = parse_open_questions(result["content"])
    return JSONResponse({
        "model": result["model"],
        "questions": questions
    })

class VerifyRequest(BaseModel):
    question: str
    options: List[str]
    correctOptionIndex: int
    userOptionIndex: int

class VerifyTFRequest(BaseModel):
    question: str
    answer: int
    optionAnswer: int

@app.post("/verify_ABCD")
async def verify_ABCD(request: VerifyRequest):
    prompt_explanation = f"""
Masz już gotowe dane pytania z geografii dla szkoły średniej:

question: "{request.question}"
options: {request.options}
correctOptionIndex: {request.correctOptionIndex}
userOptionIndex: {request.userOptionIndex}

Twoim zadaniem jest wygenerować **tylko wyjaśnienie**, dlaczego wybrany wariant (userOptionIndex) jest niepoprawny, jeśli taki był, i dlaczego poprawna odpowiedź jest właśnie ta (correctOptionIndex).  
Nie dodawaj pytania ani żadnych innych danych.  
Nie używaj Start: ani End:, po prostu zwróć czysty tekst wyjaśnienia.
"""

    explanation_text = await request_ai(prompt_explanation)
    return JSONResponse({"verify": explanation_text})

class VerifyOpenRequest(BaseModel):
    question: str
    userAnswer: str

@app.post("/verify_TRUEFALSE")
async def verify_TRUEFALSE(request: VerifyTFRequest):
    prompt_explanation = f"""
Masz już gotowe dane pytania Prawda/Fałsz z geografii dla szkoły średniej:

question: "{request.question}"
answer: {request.answer}
optionAnswer: {request.optionAnswer}

Twoim zadaniem jest wygenerować **tylko wyjaśnienie**, dlaczego wybrany wariant (optionAnswer) jest niepoprawny, jeśli taki był, i dlaczego poprawna odpowiedź jest właśnie taka (answer).  
Nie dodawaj pytania ani żadnych innych danych.  
Nie używaj Start: ani End:, po prostu zwróć czysty tekst wyjaśnienia.
"""

    explanation_text = await request_ai(prompt_explanation)
    return JSONResponse({"verify": explanation_text})


class VerifyOpenRequest(BaseModel):
    question: str
    userAnswer: str


@app.post("/verify_OPEN")
async def verify_OPEN(request: VerifyOpenRequest):
    prompt_explanation = f"""
Masz gotowe pytanie otwarte z geografii dla szkoły średniej:

question: "{request.question}"
userAnswer: "{request.userAnswer}"

Twoim zadaniem jest wygenerować **tylko wyjaśnienie lub komentarz edukacyjny** dotyczący tego pytania i odpowiedzi ucznia.  
Wyjaśnij, co było poprawne lub błędne w userAnswer, i podaj poprawny kierunek lub wskazówkę.  
Nie dodawaj pytania ani żadnych innych danych.  
Nie używaj Start: ani End:, po prostu zwróć czysty tekst wyjaśnienia.
"""

    explanation_text = await request_ai(prompt_explanation)
    return JSONResponse({"verify": explanation_text})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=True,
        timeout_keep_alive=900,
        timeout_graceful_shutdown=900
    )