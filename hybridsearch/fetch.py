"""Download paper metadata from the public arXiv API.

Usage:
    python -m hybridsearch.fetch            # 1,000 recent papers in each of five categories
"""
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
CATEGORIES = ["cs.CL", "cs.CV", "cs.DB", "cs.LG", "cs.SE"]
USER_AGENT = "arxiv-hybrid-search/0.1 (educational project)"


def parse_feed(xml_text):
    """Turn one Atom response into a list of paper dictionaries."""
    papers = []
    for entry in ET.fromstring(xml_text).findall("atom:entry", ATOM):
        primary = entry.find("arxiv:primary_category", ATOM)
        papers.append({
            "arxiv_id": entry.findtext("atom:id", "", ATOM).rsplit("/abs/", 1)[-1],
            "title": " ".join(entry.findtext("atom:title", "", ATOM).split()),
            "abstract": " ".join(entry.findtext("atom:summary", "", ATOM).split()),
            "authors": ", ".join(a.findtext("atom:name", "", ATOM) for a in entry.findall("atom:author", ATOM)),
            "published": entry.findtext("atom:published", "", ATOM)[:10],
            "category": primary.get("term") if primary is not None else "",
        })
    return papers


def fetch_category(category, n_papers=1000, page_size=200, pause=3.0):
    """Most recent papers whose primary category is the one requested.

    Cross-listed papers are skipped so every paper has exactly one label.
    arXiv asks for a three-second gap between calls.
    """
    kept, start = {}, 0
    while len(kept) < n_papers and start < n_papers * 4:
        response = requests.get(API_URL, timeout=60, headers={"User-Agent": USER_AGENT}, params={
            "search_query": f"cat:{category}", "start": start, "max_results": page_size,
            "sortBy": "submittedDate", "sortOrder": "descending"})
        response.raise_for_status()
        page = parse_feed(response.text)
        if not page:
            break
        for paper in page:
            if paper["category"] == category and paper["title"] and paper["abstract"]:
                kept.setdefault(paper["arxiv_id"], paper)
        start += page_size
        time.sleep(pause)
    return list(kept.values())[:n_papers]


def fetch_all(path="data/arxiv_papers.csv", n_per_category=1000):
    papers = []
    for category in CATEGORIES:
        found = fetch_category(category, n_per_category)
        print(f"{category}: {len(found)} papers")
        papers.extend(found)
    frame = pd.DataFrame(papers).drop_duplicates("arxiv_id").reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


if __name__ == "__main__":
    fetch_all()
