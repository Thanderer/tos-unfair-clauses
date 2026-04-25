# api.py

from fastapi import FastAPI

app = FastAPI()


@app.post("/predict")
async def predict(data: dict):
    print("\n🔥 API RECEIVED DATA")

    clauses = data.get("clauses", [])

    print("Number of clauses:", len(clauses))

    # dummy response (for now)
    results = []
    for c in clauses:
        results.append({
            "id": c.get("id"),
            "text": c.get("text"),
            "severity": "MEDIUM"
        })

    return {
        "status": "success",
        "results": results
    }