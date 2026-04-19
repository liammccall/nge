import os
import json
import requests
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.urls import path
from django.core.management import execute_from_command_line
from django.views.decorators.csrf import csrf_exempt
from llama_cpp import Llama

# -----------------------------
# Config
# -----------------------------
# HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# HF_TOKEN = os.environ.get("HF_TOKEN", "")

# # Local model (Ollama)
# LOCAL_URL = "http://localhost:11434/api/generate"
# LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:3b")

# Mode: auto | hf | local
MODE = os.environ.get("MODE", "local")

# -----------------------------
# Django config
# -----------------------------
settings.configure(
    DEBUG=True,
    SECRET_KEY="dev",
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=["*"],
    MIDDLEWARE=[],
)



# Load once (global)
llm = Llama(
    model_path="/home/liam/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=8,        # adjust for your CPU
    # n_gpu_layers=8      # 1050 Ti can handle small offload
)

def call_local(user: str, system: str):
    # tokens = llm.tokenize(prompt.encode())
    # max_input = llm.n_ctx() - 64  # reserve 512 for output
    # llm.reset()
    
    # if len(tokens) > max_input:
    #     tokens = tokens[:max_input]
    #     prompt = llm.detokenize(tokens).decode()
    
    try:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content" : system},
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=128
            )
        return {"text": output["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"text": str(e)}
# -----------------------------
# Router logic
# -----------------------------
def run_model(user_msg: str, sys_msg: str):
    # if MODE == "hf":
    #     return call_hf(prompt)

    if MODE == "local":
        return call_local(user_msg, sys_msg)

# -----------------------------
# API endpoint
# -----------------------------
@csrf_exempt
def chat_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        user_msg = data.get("user", "")
        sys_msg = data.get("system", "")
    except Exception:
        return JsonResponse({"error": "invalid json"}, status=400)

    result = run_model(user_msg,sys_msg)
    return JsonResponse(result)

def home(request):
    return HttpResponse("Django LLM router running")

# -----------------------------
# URLs
# -----------------------------
urlpatterns = [
    path("", home),
    path("api/chat", chat_api),
]

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    execute_from_command_line(["server.py", "runserver", "0.0.0.0:8000"])