from evaluator import ClaimEvaluator
from sentence_transformers import SentenceTransformer
import argparse
import pandas as pd

# model setup 

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_fn(text):
    return MODEL.encode(text)

evaluator = ClaimEvaluator(embed_fn)

# test cases 

def get_test_cases():
    return [
        {
            "name": "Strong Support",
            "claim": "The model improves accuracy by 20% on ImageNet",
            "evidence": [
                {
                    "text": "The model achieves a 20% improvement in accuracy on ImageNet benchmark.",
                    "embedding": embed_fn("The model achieves a 20% improvement in accuracy on ImageNet benchmark."),
                    "source": "https://arxiv.org"
                },
                {
                    "text": "Experimental results show gains in ImageNet classification accuracy.",
                    "embedding": embed_fn("Experimental results show gains in ImageNet classification accuracy."),
                    "source": "https://openreview.net"
                }
            ]
        },
        {
            "name": "Contradiction",
            "claim": "The model improves accuracy",
            "evidence": [
                {
                    "text": "The model decreases accuracy on standard benchmarks.",
                    "embedding": embed_fn("The model decreases accuracy on standard benchmarks."),
                    "source": "https://arxiv.org"
                }
            ]
        },
        {
            "name": "Vague Claim",
            "claim": "This model is very powerful and advanced",
            "evidence": [
                {
                    "text": "The model architecture is widely used.",
                    "embedding": embed_fn("The model architecture is widely used."),
                    "source": "https://blog.company.com"
                }
            ]
        },
        {
            "name": "Partial Support",
            "claim": "The system improves accuracy and reduces latency",
            "evidence": [
                {
                    "text": "The system improves accuracy significantly.",
                    "embedding": embed_fn("The system improves accuracy significantly."),
                    "source": "https://arxiv.org"
                }
            ]
        },
        {
            "name": "New Company",
            "claim": "Our startup model improves accuracy by 15%",
            "evidence": [
                {
                    "text": "Initial internal results show a 15% improvement in accuracy.",
                    "embedding": embed_fn("Initial internal results show a 15% improvement in accuracy."),
                    "source": "https://startup.ai"
                }
            ]
        },
        {
            "name": "Negation-Based Contradiction",
            "claim": "The model improves accuracy",
            "evidence": [
                {
                    "text": "The model does not improve accuracy in most cases.",
                    "embedding": embed_fn("The model does not improve accuracy in most cases."),
                    "source": "https://arxiv.org"
                }
            ]
        },
    ]

# runner 

def run_tests(verbose=False):
    test_cases = get_test_cases()
    rows = []

    for test in test_cases:
        result = evaluator.evaluate(test["claim"], test["evidence"])
        result["test_name"] = test["name"]
        rows.append(result)

        if verbose:
            print("\n" + "="*50)
            print(f"TEST: {test['name']}")
            print("="*50)
            for k, v in result.items():
                if isinstance(v, float):
                    print(f"{k}: {round(v, 4)}")
                else:
                    print(f"{k}: {v}")

    df = pd.DataFrame(rows)
    print("\nSummary Table:\n")
    print(df.round(4))

    return df

# CLI 

def main():
    parser = argparse.ArgumentParser(description="Claim Evaluator Test Runner")
    parser.add_argument("--verbose", action="store_true", help="Print detailed outputs")

    args = parser.parse_args()

    run_tests(verbose=args.verbose)

if __name__ == "__main__":
    main()