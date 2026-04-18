import os
import json
import requests
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.urls import path

with open("template.json") as f:
    resp = requests.post(
        "0.0.0.0:8000/api/chat",
    )