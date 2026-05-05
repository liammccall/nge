import json

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)

s1, s2, s3 = load("datatemplate1.json"), load("datatemplate2.json"), load("datatemplate3.json")
save("template1.json", s1)
save("template2.json", s2)
save("template3.json", s3)