from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    source: str
    snippet: str


class WebSearchService:
    def __init__(self, timeout: float = 1.5):
        self.timeout = timeout

    def search_exercises(self, query: str, limit: int = 3) -> list[SearchResult]:
        online = self._duckduckgo_search(f"{query} exercise recommendation ACE Mayo NHS", limit)
        return online or self._fallback_results(query, limit)

    def _duckduckgo_search(self, query: str, limit: int) -> list[SearchResult]:
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                page = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return []

        results: list[SearchResult] = []
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<href>.*?)".*?>(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            flags=re.S,
        )
        for match in pattern.finditer(page):
            target = html.unescape(match.group("href"))
            parsed_target = urllib.parse.urlparse(target)
            params = urllib.parse.parse_qs(parsed_target.query)
            real_url = params.get("uddg", [target])[0]
            title = self._clean_html(match.group("title"))
            snippet = self._clean_html(match.group("snippet"))
            source = urllib.parse.urlparse(real_url).netloc.replace("www.", "")
            if source and real_url.startswith("http"):
                results.append(SearchResult(title, real_url, source, snippet))
            if len(results) >= limit:
                break
        return results

    def _fallback_results(self, query: str, limit: int) -> list[SearchResult]:
        lowered = query.lower()
        if any(word in lowered for word in ("背", "back", "划船", "下拉")):
            results = [
                SearchResult(
                    "ACE Exercise Library",
                    "https://www.acefitness.org/resources/everyone/exercise-library/",
                    "acefitness.org",
                    "Exercise library covering strength movements such as rows and pulldowns.",
                ),
                SearchResult(
                    "Mayo Clinic Back Exercises",
                    "https://www.mayoclinic.org/healthy-lifestyle/adult-health/multimedia/back-pain/sls-20076265",
                    "mayoclinic.org",
                    "Back-friendly strengthening and mobility exercises with safety reminders.",
                ),
                SearchResult(
                    "NHS Strength And Flexibility",
                    "https://www.nhs.uk/live-well/exercise/strength-and-flexibility-exercises/",
                    "nhs.uk",
                    "Strength and flexibility exercises suitable for general fitness.",
                ),
            ]
        else:
            results = [
                SearchResult(
                    "ACE Exercise Library",
                    "https://www.acefitness.org/resources/everyone/exercise-library/",
                    "acefitness.org",
                    "Searchable exercise library for selecting movements by body area and equipment.",
                ),
                SearchResult(
                    "NHS Exercise",
                    "https://www.nhs.uk/live-well/exercise/",
                    "nhs.uk",
                    "Public health exercise guidance and beginner-friendly activity advice.",
                ),
                SearchResult(
                    "Mayo Clinic Fitness",
                    "https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/exercise/art-20048389",
                    "mayoclinic.org",
                    "General fitness guidance and safety considerations.",
                ),
            ]
        return results[:limit]

    @staticmethod
    def _clean_html(value: str) -> str:
        value = re.sub(r"<.*?>", "", value)
        return html.unescape(value).strip()
