import requests

class RAGRetriever:
    def __init__(self, base_url, auth_token=None):
        self.base_url = base_url
        self.auth_token = auth_token

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def retrieve(self, claim, extra=False):
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self._headers(),
            json={"query": claim, "top_k": 10 if extra else 5}
        )

        data = response.json()

        evidence = []
        for s in data.get("sources", []):
            evidence.append({
                "text": s["file"],  # or actual chunk text if available
                "score": s["score"],
                "chunk_index": s["chunk_index"],
                "source_type": "unknown",  # enrich later
                "timestamp": None
            })

        return evidence