import json

def extract_json(text: str):
    try:
        if "<json>" in text:
            text = text.split("<json>")[-1]
        return json.loads(text.strip())
    except Exception:
        return {"error": True}


def safe_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)