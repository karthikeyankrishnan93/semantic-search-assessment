import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
QUERY_FILE = BASE_DIR / "data" / "evaluation" / "evaluation_queries.json"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_FILE = OUTPUT_DIR / "retrieval_evaluation_report.txt"

API_URL = "http://127.0.0.1:8000/search"
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("APP_API_KEY")

def search_api(query):
    """Send one query to the Search API."""

    payload = {
        "query": query,
        "top_k": 5
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        },
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))

def get_result_field(result, field):
    """Read a result field, including nested metadata if present."""

    if field in result:
        return result[field]

    metadata = result.get("metadata", {})
    return metadata.get(field)

def main():
    if not API_KEY:
        print("ERROR: APP_API_KEY not found in .env")
        return

    if not QUERY_FILE.exists():
        print(f"ERROR: Query file not found: {QUERY_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERY_FILE, "r", encoding="utf-8") as file:
        queries = json.load(file)

    total = len(queries)
    hit_counts = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_total = 0.0
    report_lines = []

    print(f"Evaluating {total} queries...\n")

    for item in queries:
        query_id = item["id"]
        query = item["query"]
        expected_source = item["expected_source"]
        expected_page = item["expected_page"]

        try:
            response_data = search_api(query)
            results = response_data.get("results", [])

            matching_rank = 0

            for rank, result in enumerate(results, start=1):
                source = get_result_field(result, "source")
                page = get_result_field(result, "page")

                if (
                    source == expected_source
                    and str(page) == str(expected_page)
                ):
                    matching_rank = rank
                    break

            for k in (1, 3, 5):
                if matching_rank and matching_rank <= k:
                    hit_counts[k] += 1

            if matching_rank:
                reciprocal_rank_total += 1 / matching_rank

            status = (
                f"Found at rank {matching_rank}"
                if matching_rank
                else "Not found in top 5"
            )

            line = (
                f"ID {query_id}: {status} | "
                f"Expected: {expected_source}, page {expected_page}"
            )

            print(line)
            report_lines.append(line)

        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="replace")
            print(f"ID {query_id}: HTTP Error {error.code} - {message}")
            report_lines.append(
                f"ID {query_id}: HTTP Error {error.code} - {message}"
            )
            return

        except Exception as error:
            print(f"ID {query_id}: Error - {error}")
            report_lines.append(f"ID {query_id}: Error - {error}")
            return

    hit_at_1 = hit_counts[1] / total
    hit_at_3 = hit_counts[3] / total
    hit_at_5 = hit_counts[5] / total
    mrr = reciprocal_rank_total / total

    summary = [
        "",
        "========== RETRIEVAL EVALUATION SUMMARY ==========",
        f"Total queries: {total}",
        f"Hit@1: {hit_at_1:.4f} ({hit_counts[1]}/{total})",
        f"Hit@3: {hit_at_3:.4f} ({hit_counts[3]}/{total})",
        f"Hit@5: {hit_at_5:.4f} ({hit_counts[5]}/{total})",
        f"MRR: {mrr:.4f}",
    ]

    for line in summary:
        print(line)

    report_lines.extend(summary)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print(f"\nReport saved: {REPORT_FILE}")

if __name__ == "__main__":
    main()