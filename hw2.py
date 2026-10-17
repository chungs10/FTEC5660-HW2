#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any
import os
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()

def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()

def get_industry(headline):
    if not headline:
        return None
    head = headline.strip()
    if head.endswith(" Professional"):
        return head[:-len(" Professional")].strip()
    return head

def parse_mcp_output(result):
    """MCP tools return [{"type": "text", "text": "JSON"}]. Parse inner JSON."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        result = result[0].get("text", result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result

def norm(text):
    if not text:
        return ""
    return " ".join(str(text).lower().split())

def text_match(a, b):
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na == nb

def year_match(cv_year, profile_year):
    if cv_year is None or profile_year is None:
        return True
    try:
        return int(cv_year) == int(profile_year)
    except (ValueError, TypeError):
        return True

def python_compare(cv_extract, linkedin, cv_file, candidate_id):
    prefix = f"[{cv_file}] candidate {candidate_id}:" if candidate_id else f"[{cv_file}]:"
    discrepancies = []

    if not text_match(cv_extract.get("name"), linkedin.get("name")):
        print(f"{prefix} Name discrepancy: CV '{cv_extract.get('name')}' vs LinkedIn '{linkedin.get('name')}'")
        discrepancies.append("name")

    if not text_match(cv_extract.get("city"), linkedin.get("city")):
        print(f"{prefix} City discrepancy: CV '{cv_extract.get('city')}' vs LinkedIn '{linkedin.get('city')}'")
        discrepancies.append("city")

    cv_experience = cv_extract.get("experience") or []
    linkedin_experience = linkedin.get("experience") or []
    for cv_job in cv_experience:
        cv_company = cv_job.get("company", "")
        linkedin_match = None
        for linked_job in linkedin_experience:
            if text_match(cv_company, linked_job.get("company", "")):
                linkedin_match = linked_job
                break
        if not linkedin_match:
            linkedin_companies = [j.get("company", "") for j in linkedin_experience]
            print(f"{prefix} Employer discrepancy: CV '{cv_company}' vs LinkedIn {linkedin_companies}")
            discrepancies.append("employer")
            continue
        if not year_match(cv_job.get("start_year"), linkedin_match.get("start_year")):
            print(f"{prefix} Start year discrepancy at '{cv_company}': CV {cv_job.get('start_year')} vs LinkedIn {linkedin_match.get('start_year')}")
            discrepancies.append("start_year")
        if not year_match(cv_job.get("end_year"), linkedin_match.get("end_year")):
            print(f"{prefix} End year discrepancy at '{cv_company}': CV {cv_job.get('end_year')} vs LinkedIn {linkedin_match.get('end_year')}")
            discrepancies.append("end_year")

    cv_education = cv_extract.get("education") or []
    linkedin_education = linkedin.get("education") or []
    for cv_edu in cv_education:
        cv_school = cv_edu.get("school", "")
        linkedin_match = None
        for linked_edu in linkedin_education:
            if text_match(cv_school, linked_edu.get("school", "")):
                linkedin_match = linked_edu
                break
        if not linkedin_match:
            linkedin_schools = [e.get("school", "") for e in linkedin_education]
            print(f"{prefix} School discrepancy: CV '{cv_school}' vs LinkedIn {linkedin_schools}")
            discrepancies.append("school")
            continue
        if not year_match(cv_edu.get("end_year"), linkedin_match.get("end_year")):
            print(f"{prefix} Grad year discrepancy at '{cv_school}': CV {cv_edu.get('end_year')} vs LinkedIn {linkedin_match.get('end_year')}")
            discrepancies.append("grad_year")

    return discrepancies

def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    extract_from_cv_PROMPT = """
            You are reading a CV.
        
            Return ONLY a JSON object with exactly these keys:
           {{
            "name": "string",
            "city": "string (city only, no country)",
            "country": "string or null",
            "headline": "string or null",
            "experience": [
                {{
                "company": "string",
                "title": "string (full title as written)",
                "seniority": "junior | mid | senior | null",
                "start_year": integer or null,
                "end_year": integer or null,
                "is_current": boolean
                }}
            ],
            "education": [
                {{
                "school": "string",
                "degree": "string",
                "field": "string",
                "end_year": integer or null
                }}
            ],
            "skills": [
                {{ "name": "string" }}
            ]
            }}
            RULES:
            - Use exact key names. Do NOT rename.
            - "city" is the city name only. "country" is separate.
            - Years are integers. If "Present", end_year=null and is_current=true.
            - "title" is the FULL title. Also extract "seniority" separately.
            - "headline" is the line below the name. If missing, null.
            - "skills" entries are objects, each with a single "name" key.
            - Ignore descriptions, bullets, hometown.
            - Return ONLY JSON.
            - CRITICAL: If the CV title has no seniority word, "seniority" MUST be null.
            - "education.end_year" is the graduation year.
            - Missing scalar fields (name, city, country, headline): use null.
            - Empty list fields (experience, education, skills): use [].
            Do NOT guess. Do NOT infer from the title. Do NOT copy a value from elsewhere.

            CV:
            {cv_text}

            Return ONLY the JSON. No explanation. No markdown fences.
            """
    compare_cv_with_linkedin_PROMPT = """   
    You are comparing a CV against a LinkedIn profile's judgment fields.

        CV fields:
        - experience: {cv_experience}
        - education: {cv_education}
        - skills: {cv_skills}

        Linkedin fields:
        - experience: {linkedin_experience}
        - education: {linkedin_education}
        - skills: {linkedin_skills}

        Rules:
        - "Senior Manager" on the CV vs "Manager" with seniority=senior on the profile -> NOT a discrepancy.
        - "Senior Manager" on the CV vs "Manager" with seniority=junior on the profile -> IS a discrepancy.
        - "BSc" vs "Bachelor of Science" -> NOT a discrepancy (same degree, different wording).
        - "UI/UX Design" vs "UI/UX" -> NOT a discrepancy (same skill, different wording).
        - Job start_year or end_year differing between CV and profile -> IS a discrepancy.
        - Education end_year (graduation year) differing between CV and profile -> IS a discrepancy.
        - If one side has a null year and the other has a value, do NOT flag — missing data is not a discrepancy.
        - Compare skills by the "name" field on both sides. Ignore "proficiency".
        - A CV skill whose name is not in the profile's skill names -> IS a discrepancy.
        - Fewer skills on the CV than the profile -> NOT a discrepancy.
        Seniority rule:
        - If the CV seniority is null, do NOT flag a seniority discrepancy.
        A CV that omits seniority is being terse, not lying.
        - Only flag a seniority discrepancy when BOTH sides state a seniority AND they differ.

        Return ONLY JSON:
        {{"discrepancies": ["description 1", "description 2"]}}

        If no discrepancies: {{"discrepancies": []}}
        """
    ### YOUR CODE HERE
    llm = ChatDeepSeek(
        model=MODEL_NAME,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.0,
    )
    parser = JsonOutputParser()

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", extract_from_cv_PROMPT),
    ])
    compare_prompt = ChatPromptTemplate.from_messages([
        ("system", compare_cv_with_linkedin_PROMPT),
    ])

    mcp_tools = {t.name: t for t in tools}

    return {
        "extract_from_cv": extract_prompt | llm,
        "compare_cv_linkedin": compare_prompt | llm,
        "search_linkedin": mcp_tools["search_linkedin_people"],
        "fetch_linkedin": mcp_tools["get_linkedin_profile"],
        "parser":parser
    }


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    ### YOUR CODE HERE
    sem = asyncio.Semaphore(3)

    async def one_cv(cv_file, cv_text):
        async with sem:
            try:
                #Grabbed data from CV and parsed it
                extract_cv = await agent["extract_from_cv"].ainvoke({"cv_text": cv_text})
                parsed_extract_from_cv = agent["parser"].parse(response_text(extract_cv))

                
                #search using name, city, and industry
                cv_name = parsed_extract_from_cv.get("name","")
                cv_city = parsed_extract_from_cv.get("city","")
                cv_industry = get_industry(parsed_extract_from_cv.get("headline",""))

                if not cv_name:
                    print(f"[{cv_file}] has no name, That's an automatic zero")
                    return cv_file, 0.0

                
                search_link = {"q": cv_name}
                if cv_city:
                    search_link["location"] = cv_city
                if cv_industry:
                    search_link["industry"] = cv_industry


                search_linkedin = await agent["search_linkedin"].ainvoke(search_link)
                parsed_search_linkedin= parse_mcp_output(search_linkedin)

                if not parsed_search_linkedin:
                    print(f"[{cv_file}] has been searched on Linkedin and nothing returned so that's another automatic zero")
                    return cv_file, 0.0

                profiles = []
                for profile in parsed_search_linkedin:
                    pid = profile.get("id")
                    if not pid:
                        continue
                    full_p = await agent["fetch_linkedin"].ainvoke({"person_id": pid})
                    parsed_full_p = parse_mcp_output(full_p)
                    if isinstance(parsed_full_p , dict) and "error" not in parsed_full_p :
                        print(f"[{cv_file}] added candidate ID: {pid}")
                        profiles.append(parsed_full_p )

                print(f"[{cv_file}] {len(profiles)} candidates found on LinkedIn, let's see what we can find.")
                for candidate in profiles:
                    if python_compare(parsed_extract_from_cv, candidate, cv_file, candidate.get("id")):
                        continue

                    llm_compare = {
                        "cv_experience": json.dumps(parsed_extract_from_cv.get("experience", [])),
                        "cv_education": json.dumps(parsed_extract_from_cv.get("education", [])),
                        "cv_skills": json.dumps(parsed_extract_from_cv.get("skills", [])),
                        "linkedin_experience": json.dumps(candidate.get("experience", [])),
                        "linkedin_education": json.dumps(candidate.get("education", [])),
                        "linkedin_skills": json.dumps(candidate.get("skills", [])),
                    }
                    extract_llm = await agent["compare_cv_linkedin"].ainvoke(llm_compare)
                    parsed_extract_llm = agent["parser"].parse(response_text(extract_llm))
                    if parsed_extract_llm.get("discrepancies"):
                        print(f"[{cv_file}] candidate {candidate.get('id')} failed LLM compare: {parsed_extract_llm['discrepancies']}")
                        continue
                    if not parsed_extract_llm.get("discrepancies"):
                        print(f"[{cv_file}] has found a match and therefore this file has been accepted")
                        return cv_file, 1.0
                
                print(f"[{cv_file}] didn't pass the judgement and discrepancies have been found")
                return cv_file, 0.0

            except Exception as e:
                print(f"[{cv_file}] has caused an error and will return a 0.0")
                return cv_file, 0.0

    tasks = [one_cv(f, t) for f, t in cvs.items()]
    return dict(await asyncio.gather(*tasks))



# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())