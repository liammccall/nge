import json
import requests
import random

MAX_ITERATIONS = 1000

def chat(user: str, system: str) -> str:
    prompt=f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""


    resp = requests.post(
        "http://localhost:8000/api/chat",
        json={"user": user, "system": system}   # IMPORTANT: send "prompt", not system/user split
    )
    resp.raise_for_status()
    return resp.json().get("text", "")

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)

CATEGORY_THEORY_AXIOM = (
    "a Category is a collection of objects, where for each pair a, b "
    "there is a set of morphisms c(a, b). There is a binary composition "
    "operation between these sets which is associative and satisfies the "
    "left and right unit laws."
)

sys = load("system.json")


last_prompts = []

for i in range(MAX_ITERATIONS):
    t1, t2, t3 = load("template1.json"), load("template2.json"), load("template3.json")
    current_prompt: str = t3["user"]
    
    mut_string = " "

    if last_prompts and random.random() < 0.1:
        sentences = [s.strip() for s in current_prompt.split(mut_string) if s.strip()]
        new_sentences = []

        for sentence in sentences:
            if random.random() < 0.1:
                donor_prompt = random.choice(last_prompts)
                donor_sentences = [s.strip() for s in donor_prompt.split(mut_string) if s.strip()]
                
                if donor_sentences:
                    new_sentences.append(random.choice(donor_sentences))
                else:
                    new_sentences.append(sentence)
            else:
                new_sentences.append(sentence)

        current_prompt = mut_string.join(new_sentences) + "."

    last_prompts.append(current_prompt)

    # Request 1: Generate a response to the current prompt
    if i == 0:
        response_text = t1["user"]  # Use pre-seeded response on first pass
    else:
        response_text = chat(
            f"{current_prompt}",
            sys["query1"])
        save("template1.json", {"user": response_text})

    # Request 2: Evaluate via category theory lens
    evaluation = chat(
        f"Provide a brief (few sentences) statement and evaluation of the following theory: {response_text}",
        sys["query2"]
    )
    save("template2.json", {"user": evaluation})

    # Request 3: Synthesize a better prompt
    improved_prompt = chat(
    sys["query3"],
    f"""Original Prompt:
{current_prompt}

Response:
{response_text}

Evaluation:
{evaluation}"""
)
    save("template3.json", {"user": improved_prompt})

    print(f"[Iter {i+1}] Prompt: {improved_prompt[:120]}...")