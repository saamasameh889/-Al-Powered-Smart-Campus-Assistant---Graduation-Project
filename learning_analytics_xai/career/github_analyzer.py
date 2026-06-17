"""
github_analyzer.py — GitHub portfolio analyzer for Product E (Career Advisor)
===============================================================================
API calls per analysis (no token required for public profiles):
  GET /users/{username}                    → profile stats
  GET /users/{username}/repos (paginated)  → all public repos
  GET /users/{username}/events (3 pages)   → recent activity (90 days)
  GET /repos/{owner}/{repo}/languages      → per top-6 repo, parallel

Programme alignment uses TWO language sets:
  • all_languages: primary language field across ALL repos (fast, comprehensive)
  • lang_pct: detailed byte-level breakdown from top-6 repos (for display)
Alignment score is bidirectional:
  • Core coverage  (60%): % of must-have programme languages present
  • Full relevance (40%): % of the student's languages that are programme-relevant
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import requests

# ── Programme core languages (must-haves shown in gap warnings) ───────────────
# Kept intentionally small — these drive the gap message, not the score.
PROGRAMME_CORE: dict[str, list[str]] = {
    # CSAI is a superset major (CS + AI + SWE + DSAI) — core is deliberately
    # minimal so that any CS-relevant work scores well.
    "CSAI": ["Python", "C++"],
    "DSAI": ["Python", "R", "Jupyter Notebook", "SQL"],
    "SWE":  ["JavaScript", "TypeScript", "Java"],
    "EEE":  ["C", "C++", "MATLAB", "Verilog"],
    "MECH": ["MATLAB", "C++", "Python"],
    "CIV":  ["Python", "MATLAB", "R"],
    "MATH": ["Python", "R", "MATLAB", "Jupyter Notebook"],
    "PHYS": ["Python", "C++", "MATLAB"],
    "CHEM": ["Python", "R", "MATLAB"],
    "BUS":  ["Python", "SQL", "R"],
    "FIN":  ["Python", "SQL", "R"],
}

# ── Full programme stack (all relevant languages + tools) ─────────────────────
# CSAI is explicitly the union of DSAI + SWE + EEE-systems + AI-infra:
#   any competent CS/SWE/DSAI/DevOps language is valid CSAI work.
# GitHub detects CSS preprocessors (SCSS/Sass/Less), template languages (MDX,
# Nunjucks), and build tools (Makefile, CMake) as separate "languages" — include
# them so profiles aren't penalised for common tooling.
_SWE_LANGS  = [
    "JavaScript", "TypeScript", "Java", "Python", "Go", "Rust",
    "C#", "Kotlin", "Swift", "Dart", "Ruby", "PHP", "Elixir",
    "Scala", "HTML", "CSS", "SCSS", "Sass", "Less",
    "Vue", "Svelte", "Astro", "Handlebars", "Nunjucks", "MDX",
    "C", "C++", "Shell", "Dockerfile", "Makefile",
]
_DSAI_LANGS = [
    "Python", "R", "Jupyter Notebook", "SQL", "Scala", "Julia",
    "MATLAB", "Shell", "JavaScript", "TypeScript", "Go", "Java",
    "Dockerfile", "Haskell",
]
_SYS_LANGS  = [
    "C", "C++", "Rust", "Go", "Assembly", "Fortran", "CUDA",
    "CMake", "Makefile", "Zig", "Nim", "D",
]

PROGRAMME_STACK: dict[str, list[str]] = {
    # CSAI = everything: SWE ∪ DSAI ∪ systems ∪ AI-infra
    "CSAI": sorted(set(
        _SWE_LANGS + _DSAI_LANGS + _SYS_LANGS + [
            "Nix", "Perl", "Tcl", "Lua", "Groovy",
            "PowerShell", "Batchfile", "HCL", "Bicep",
        ]
    )),
    "DSAI": _DSAI_LANGS + [
        "C++", "Rust", "Julia", "MATLAB", "CUDA", "Dockerfile",
    ],
    "SWE": _SWE_LANGS + [
        "Ruby", "PHP", "Dart", "Kotlin", "Swift",
        "GraphQL", "PLpgSQL", "Nix",
    ],
    "EEE": [
        "C", "C++", "Python", "MATLAB", "Verilog", "SystemVerilog",
        "VHDL", "Assembly", "Rust", "Julia", "Shell", "Fortran",
    ],
    "MECH": [
        "MATLAB", "C++", "Python", "C", "Fortran", "Julia",
        "Shell", "Rust",
    ],
    "CIV": [
        "Python", "MATLAB", "R", "C++", "Julia", "Fortran",
        "Jupyter Notebook", "SQL",
    ],
    "MATH": [
        "Python", "R", "MATLAB", "Julia", "C++", "Haskell",
        "Jupyter Notebook", "Fortran", "Lean", "Coq",
    ],
    "PHYS": [
        "Python", "C++", "MATLAB", "Fortran", "Julia", "C",
        "Jupyter Notebook", "Assembly", "Rust",
    ],
    "CHEM": [
        "Python", "R", "MATLAB", "Julia", "C++", "Fortran",
        "Jupyter Notebook",
    ],
    "BUS":  ["Python", "SQL", "R", "JavaScript", "TypeScript", "VBA", "Jupyter Notebook", "Shell"],
    "FIN":  ["Python", "SQL", "R", "C++", "Julia", "VBA", "JavaScript", "TypeScript", "MATLAB", "Scala"],
}

# ── Domain keyword → topic/name matching ─────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ML/AI":    ["machine-learning", "deep-learning", "neural", "tensorflow", "pytorch",
                 "keras", "nlp", "computer-vision", "reinforcement", "llm", "gpt",
                 "transformers", "classification", "regression", "cnn", "rnn", "lstm",
                 "ai", "ml", "yolo", "diffusion", "generative", "rag", "embedding"],
    "Web":      ["web", "frontend", "backend", "react", "angular", "vue", "django",
                 "flask", "fastapi", "node", "express", "rest", "graphql", "html",
                 "css", "nextjs", "svelte", "nuxt", "remix", "astro"],
    "Data":     ["data", "analysis", "visualization", "pandas", "numpy", "sql", "etl",
                 "dashboard", "analytics", "bi", "tableau", "powerbi", "spark",
                 "hadoop", "pipeline", "warehouse", "dbt", "airflow"],
    "Systems":  ["os", "kernel", "compiler", "embedded", "iot", "rtos", "driver",
                 "hardware", "assembly", "microcontroller", "arduino", "raspberry",
                 "fpga", "verilog", "systems", "runtime", "wasm", "bytecode"],
    "Security": ["security", "crypto", "ctf", "penetration", "vulnerability", "malware",
                 "hacking", "reverse", "forensics", "firewall", "exploit", "audit"],
    "Mobile":   ["android", "ios", "flutter", "react-native", "mobile", "app",
                 "kotlin", "swift", "expo"],
    "Game":     ["game", "unity", "unreal", "pygame", "opengl", "vulkan", "directx",
                 "godot", "rendering", "shader"],
    "DevOps":   ["docker", "kubernetes", "ci", "cd", "deployment", "terraform",
                 "ansible", "github-actions", "jenkins", "aws", "gcp", "azure",
                 "cloud", "helm", "k8s", "infrastructure"],
    "Research": ["paper", "research", "thesis", "simulation", "algorithm",
                 "optimization", "numerical", "scientific", "experiment", "benchmark"],
    "Tooling":  ["cli", "tool", "library", "sdk", "plugin", "extension", "bundler",
                 "compiler", "linter", "formatter", "parser", "transpiler"],
}


class GitHubAnalyzer:
    BASE    = "https://api.github.com"
    TIMEOUT = 10

    def __init__(self, token: str | None = None):
        self.headers = {
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    # ── Raw API helpers ────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict | None = None) -> Any:
        r = requests.get(url, headers=self.headers, params=params, timeout=self.TIMEOUT)
        if r.status_code == 404:
            raise ValueError(f"GitHub user/repo not found: {url}")
        if r.status_code == 403:
            reset_ts = r.headers.get("X-RateLimit-Reset", "")
            try:
                reset_str = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc).strftime("%H:%M UTC")
            except Exception:
                reset_str = reset_ts or "unknown"
            raise RuntimeError(
                f"GitHub rate limit hit (60 req/hr unauthenticated). "
                f"Resets at {reset_str}. Add a GitHub token in the field above to get 5,000 req/hr."
            )
        r.raise_for_status()
        return r.json()

    def fetch_profile(self, username: str) -> dict:
        return self._get(f"{self.BASE}/users/{username}")

    def fetch_repos(self, username: str) -> list[dict]:
        all_repos, page = [], 1
        while True:
            chunk = self._get(
                f"{self.BASE}/users/{username}/repos",
                params={"per_page": 100, "page": page, "sort": "updated"},
            )
            if not chunk:
                break
            all_repos.extend(chunk)
            if len(chunk) < 100:
                break
            page += 1
        return all_repos

    def fetch_events(self, username: str, is_org: bool = False) -> list[dict]:
        """Fetch up to 300 events (3 pages). Uses org endpoint for organizations."""
        endpoint = (
            f"{self.BASE}/orgs/{username}/events"
            if is_org else
            f"{self.BASE}/users/{username}/events/public"
        )
        all_events, page = [], 1
        while page <= 3:
            try:
                chunk = self._get(endpoint, params={"per_page": 100, "page": page})
            except Exception:
                break
            if not chunk:
                break
            all_events.extend(chunk)
            if len(chunk) < 100:
                break
            page += 1
        return all_events

    def fetch_languages(self, owner: str, repo: str) -> dict[str, int]:
        try:
            return self._get(f"{self.BASE}/repos/{owner}/{repo}/languages")
        except Exception:
            return {}

    def _has_profile_readme(self, username: str) -> bool:
        try:
            self._get(f"{self.BASE}/repos/{username}/{username}")
            return True
        except Exception:
            return False

    # ── Metric helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _repo_score(r: dict, now_ts: float) -> float:
        stars  = r.get("stargazers_count", 0)
        pushed = r.get("pushed_at") or r.get("updated_at") or ""
        try:
            ts       = datetime.fromisoformat(pushed.replace("Z", "+00:00")).timestamp()
            age_days = (now_ts - ts) / 86400
            recency  = max(0.0, 1.0 - age_days / 365)
        except Exception:
            recency = 0.0
        return stars * 2 + recency * 5

    @staticmethod
    def _quality_score(r: dict) -> int:
        """0-4: non-empty original repo, description, topics, license."""
        return (
            int(bool(r.get("has_wiki") is not None and not r.get("fork") and r.get("size", 0) > 0))
            + int(bool((r.get("description") or "").strip()))
            + int(bool(r.get("topics")))
            + int(bool(r.get("license")))
        )

    @staticmethod
    def _classify_domain(name: str, desc: str, topics: list[str]) -> list[str]:
        text = " ".join([name.lower(), (desc or "").lower()] + [t.lower() for t in topics])
        found = [d for d, kws in DOMAIN_KEYWORDS.items() if any(kw in text for kw in kws)]
        return found or ["General"]

    @staticmethod
    def _activity_from_events(events: list[dict]) -> dict:
        now    = datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - 90 * 86400
        weekly_buckets: dict[int, int] = {}
        collab_count = 0
        push_ts_list: list[float] = []

        COLLAB_TYPES = {
            "ForkEvent", "PullRequestEvent", "PullRequestReviewEvent",
            "IssuesEvent", "IssueCommentEvent", "PullRequestReviewCommentEvent",
            "CommitCommentEvent", "CreateEvent", "ReleaseEvent",
        }

        for ev in events:
            try:
                ts = datetime.fromisoformat(
                    ev.get("created_at", "").replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                continue
            if ts < cutoff:
                continue

            ev_type = ev.get("type", "")
            if ev_type == "PushEvent":
                payload = ev.get("payload", {})
                # payload.size gives true commit count; commits list is capped at 20
                n = payload.get("size") or len(payload.get("commits", [])) or 1
                week = int((now.timestamp() - ts) / (7 * 86400))
                weekly_buckets[week] = weekly_buckets.get(week, 0) + n
                push_ts_list.append(ts)
            elif ev_type in COLLAB_TYPES:
                collab_count += 1

        total    = sum(weekly_buckets.values())
        cpw      = round(total / 13, 2) if total else 0.0
        last_ts  = max(push_ts_list) if push_ts_list else None

        return {
            "total_commits_90d": total,
            "commits_per_week":  cpw,
            "active_weeks":      len(weekly_buckets),
            "collab_events":     collab_count,
            "_last_push_ts":     last_ts,
        }

    @staticmethod
    def _language_breakdown(lang_maps: list[dict[str, int]]) -> dict[str, float]:
        totals: dict[str, int] = {}
        for lm in lang_maps:
            for lang, b in lm.items():
                totals[lang] = totals.get(lang, 0) + b
        grand = sum(totals.values()) or 1
        return {k: round(v / grand * 100, 1)
                for k, v in sorted(totals.items(), key=lambda x: -x[1])}

    @staticmethod
    def _programme_alignment(all_languages: set[str],
                              lang_pct: dict[str, float],
                              programme: str) -> dict:
        """
        Bidirectional alignment score:
          • Core coverage (60%): % of must-have programme languages found
          • Relevance     (40%): % of student's own languages that are programme-relevant
        """
        core     = PROGRAMME_CORE.get(programme, [])
        extended = PROGRAMME_STACK.get(programme, [])
        extended_set = set(extended)

        # Combine all known languages (primary lang from all repos + detailed from top 6)
        known = all_languages | set(lang_pct.keys())

        matched_core    = [l for l in core     if l in known]
        missing_core    = [l for l in core     if l not in known]
        matched_ext     = [l for l in extended if l in known]

        core_coverage = round(len(matched_core) / max(len(core), 1) * 100)
        relevance     = round(
            len([l for l in known if l in extended_set]) / max(len(known), 1) * 100
        )
        # Relevance (70%) dominates: a profile doing CSAI-relevant work should score
        # well even if it hasn't covered every core language yet.
        # Core coverage (30%) nudges score down when must-haves are completely absent.
        score = round(relevance * 0.70 + core_coverage * 0.30)

        return {
            "score":        score,
            "core_coverage": core_coverage,
            "relevance":    relevance,
            "matched":      matched_ext,
            "matched_core": matched_core,
            "missing":      missing_core,   # only missing CORE shown in gaps
            "expected":     core,
        }

    def _detect_gaps(self, repos: list[dict], activity: dict,
                     all_languages: set[str], lang_pct: dict[str, float],
                     alignment: dict, profile: dict,
                     has_profile_readme: bool, programme: str,
                     is_org: bool = False) -> list[str]:
        gaps = []
        n = max(len(repos), 1)

        # Repo metadata quality
        no_desc = [r["name"] for r in repos if not (r.get("description") or "").strip()]
        if no_desc:
            gaps.append(f"{len(no_desc)}/{n} repos have no description (e.g. {', '.join(no_desc[:3])})")

        no_topics = [r["name"] for r in repos if not r.get("topics")]
        if len(no_topics) > n * 0.5:
            gaps.append(f"{len(no_topics)}/{n} repos have no topics/tags — topics improve SEO and recruiter discovery")

        no_license = [r["name"] for r in repos if not r.get("license")]
        if len(no_license) > n * 0.6:
            gaps.append(f"{len(no_license)} repos missing a license — unlicensed code cannot legally be used or contributed to")

        # Activity
        if activity["days_since_push"] > 60:
            gaps.append(f"Last push was {activity['days_since_push']} days ago — profile appears inactive to recruiters")
        if activity["commits_per_week"] < 1.0:
            gaps.append(f"Low commit frequency ({activity['commits_per_week']:.1f}/week) — daily contributions signal sustained learning")
        if activity["collab_events"] == 0:
            gaps.append("No collaborative activity detected — open-source contributions (PRs, reviews) are a top hiring signal")

        # Stack alignment — only flag if core coverage is low
        if alignment["core_coverage"] < 40 and alignment["missing"]:
            gaps.append(
                f"Missing core {programme} languages: {', '.join(alignment['missing'][:4])} "
                f"— these appear in most {programme} job descriptions"
            )

        # Personal profile quality (skip for orgs)
        if not is_org:
            if not has_profile_readme:
                gaps.append("No profile README — a pinned README is the single highest-ROI portfolio improvement")
            if not profile.get("bio"):
                gaps.append("Empty GitHub bio — a 1-2 line bio with your stack and interests improves recruiter click-through")

        # Fork-heavy profile
        fork_count = sum(1 for r in repos if r.get("fork"))
        if fork_count > n * 0.6:
            gaps.append(f"{fork_count}/{n} repos are forks — original projects demonstrate initiative, not just consumption")

        return gaps

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyze(self, username: str, programme: str = "CSAI",
                semester: int = 4) -> dict:
        now_ts = datetime.now(tz=timezone.utc).timestamp()

        # Profile first — needed to detect org vs user (changes events endpoint)
        profile = self.fetch_profile(username)
        is_org  = profile.get("type", "User") == "Organization"

        # Parallel: repos + events + profile-README
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_repos  = ex.submit(self.fetch_repos,         username)
            f_events = ex.submit(self.fetch_events,        username, is_org)
            f_readme = ex.submit(self._has_profile_readme, username)

        repos_raw          = f_repos.result()
        events             = f_events.result()
        has_profile_readme = f_readme.result() if not is_org else False

        # Top 6 repos for detailed language breakdown display
        scored = sorted(repos_raw, key=lambda r: self._repo_score(r, now_ts), reverse=True)
        top6   = scored[:6]

        # Parallel language breakdown for top 6
        lang_maps: list[dict[str, int]] = [{} for _ in top6]
        with ThreadPoolExecutor(max_workers=6) as ex:
            fut_map = {
                ex.submit(self.fetch_languages, username, r["name"]): i
                for i, r in enumerate(top6)
            }
            for fut in as_completed(fut_map):
                lang_maps[fut_map[fut]] = fut.result()

        # Detailed language % (top 6 repos, byte-level)
        lang_pct = self._language_breakdown(lang_maps)

        # All languages: primary language field across EVERY repo (no extra API calls)
        # This is used for alignment — more representative than just top 6
        all_languages: set[str] = {
            r["language"] for r in repos_raw if r.get("language")
        }

        # Activity metrics
        activity = self._activity_from_events(events)

        # days_since_push: repos pushed_at is authoritative (events window is limited)
        repo_push_ts = []
        for r in repos_raw:
            pushed = r.get("pushed_at") or ""
            try:
                repo_push_ts.append(
                    datetime.fromisoformat(pushed.replace("Z", "+00:00")).timestamp()
                )
            except Exception:
                pass
        if repo_push_ts:
            days_since_push = int((now_ts - max(repo_push_ts)) / 86400)
        elif activity.get("_last_push_ts"):
            days_since_push = int((now_ts - activity["_last_push_ts"]) / 86400)
        else:
            days_since_push = 999
        activity["days_since_push"] = days_since_push
        activity.pop("_last_push_ts", None)

        # Programme alignment (bidirectional)
        alignment = self._programme_alignment(all_languages, lang_pct, programme)

        # Per-repo metadata for top 6
        top_repos = []
        for i, r in enumerate(top6):
            lm = lang_maps[i]
            total_b = sum(lm.values()) or 1
            repo_langs = {k: round(v / total_b * 100)
                          for k, v in sorted(lm.items(), key=lambda x: -x[1])[:4]}
            top_repos.append({
                "name":        r["name"],
                "description": (r.get("description") or "").strip() or "(no description)",
                "stars":       r.get("stargazers_count", 0),
                "forks_count": r.get("forks_count", 0),
                "languages":   repo_langs,
                "topics":      r.get("topics") or [],
                "quality":     self._quality_score(r),
                "pushed_at":   r.get("pushed_at", ""),
                "url":         r.get("html_url", ""),
                "is_fork":     r.get("fork", False),
                "domains":     self._classify_domain(
                    r["name"], r.get("description") or "", r.get("topics") or []
                ),
            })

        # Aggregate domains across ALL repos
        all_domains: dict[str, int] = {}
        for r in repos_raw:
            for d in self._classify_domain(
                r["name"], r.get("description") or "", r.get("topics") or []
            ):
                all_domains[d] = all_domains.get(d, 0) + 1
        top_domains = sorted(all_domains, key=lambda k: -all_domains[k])[:5]

        # Aggregate stats across ALL repos
        total_stars  = sum(r.get("stargazers_count", 0) for r in repos_raw)
        total_forks  = sum(r.get("forks_count",       0) for r in repos_raw)
        total_issues = sum(r.get("open_issues_count",  0) for r in repos_raw)
        avg_quality  = sum(self._quality_score(r) for r in top_repos) / max(len(top_repos), 1)

        gaps = self._detect_gaps(
            repos_raw, activity, all_languages, lang_pct, alignment,
            profile, has_profile_readme, programme, is_org=is_org,
        )

        joined_year = profile.get("created_at", "")[:4] or "?"
        presentation_score = (
            int(has_profile_readme)
            + int(bool(profile.get("bio")))
            + int(bool(
                profile.get("avatar_url")
                and "gravatar" not in profile.get("avatar_url", "")
            ))
        )

        analysis = {
            "username":           username,
            "name":               profile.get("name") or username,
            "is_org":             is_org,
            "programme":          programme,
            "semester":           semester,
            "joined_year":        joined_year,
            "bio":                profile.get("bio") or "",
            "location":           profile.get("location") or "",
            "followers":          profile.get("followers", 0),
            "following":          profile.get("following", 0),
            "n_repos":            profile.get("public_repos", len(repos_raw)),
            "lang_pct":           lang_pct,
            "all_languages":      sorted(all_languages),
            "activity":           activity,
            "top_repos":          top_repos,
            "total_stars":        total_stars,
            "total_forks":        total_forks,
            "total_open_issues":  total_issues,
            "avg_quality":        round(avg_quality, 2),
            "top_domains":        top_domains,
            "alignment":          alignment,
            "has_profile_readme": has_profile_readme,
            "presentation_score": presentation_score,
            "gaps":               gaps,
        }

        analysis["prompt"] = self.build_prompt(analysis)
        return analysis

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def build_prompt(self, a: dict) -> str:
        lang_str  = ", ".join(f"{l} {p}%" for l, p in list(a["lang_pct"].items())[:8]) \
                    or "no detailed language data"
        all_langs = ", ".join(a["all_languages"][:15]) or "unknown"

        repo_lines = []
        for r in a["top_repos"]:
            lang_part = ", ".join(f"{l}:{p}%" for l, p in r["languages"].items()) or "?"
            topics    = " ".join(f"#{t}" for t in r["topics"][:5]) or "(no topics)"
            fork_tag  = " [fork]" if r["is_fork"] else ""
            repo_lines.append(
                f"  • {r['name']}{fork_tag}  ⭐{r['stars']}  🍴{r['forks_count']}"
                f"  [quality {r['quality']}/4]  [{lang_part}]  {topics}\n"
                f"    {r['description'][:130]}"
            )

        gaps_str = "\n".join(f"  ⚠ {g}" for g in a["gaps"]) or "  ✓ No major gaps"

        al = a["alignment"]
        matched_str = ", ".join(al["matched"][:8]) or "none"
        missing_str = ", ".join(al["missing"][:5]) or "none"

        account_type = "Organization" if a.get("is_org") else "Student"

        return f"""You are an expert GitHub portfolio auditor and tech industry HR advisor. \
You evaluate developer portfolios with the same rigour as a senior hiring manager at a \
top-tier tech company (Google, Meta, Stripe, or a well-funded startup). \
Your job is to give honest, specific, industry-aware feedback that will meaningfully \
improve this developer's chances of landing a competitive internship or job.

═══════════════════════════════════════════════════════════════════
PROFILE: @{a["username"]}  ({account_type})  |  {a["programme"]} — Semester {a["semester"]} of 8
Account created: {a["joined_year"]}  |  Bio: {a["bio"] or "(empty)"}
Location: {a["location"] or "not specified"}
Followers: {a["followers"]:,}  |  Public repos: {a["n_repos"]}
═══════════════════════════════════════════════════════════════════

PORTFOLIO METRICS (all {a["n_repos"]} repos scanned):
  ⭐ Total stars      : {a["total_stars"]:,}
  🍴 Total forks      : {a["total_forks"]:,}   (how many times others forked YOUR repos)
  🐛 Open issues      : {a["total_open_issues"]:,}
  📊 Domain coverage  : {", ".join(a["top_domains"])}
  🔧 Repo quality avg : {a["avg_quality"]:.1f}/4  (description + topics + license + non-empty)
  🎭 Presentation     : {a["presentation_score"]}/3  (profile README, bio, custom avatar)

LANGUAGE DISTRIBUTION:
  All languages used (all {a["n_repos"]} repos, primary lang): {all_langs}
  Detailed breakdown (top 6 repos by byte count): {lang_str}

ACTIVITY — LAST 90 DAYS (events API, up to 300 events):
  Commits tracked    : {a["activity"]["total_commits_90d"]:,}
  Commits / week     : {a["activity"]["commits_per_week"]:.1f}
  Active weeks       : {a["activity"]["active_weeks"]} / 13
  Collaborative acts : {a["activity"]["collab_events"]}  (PRs, PR reviews, issues, releases, creates)
  Days since push    : {a["activity"]["days_since_push"]}

{a["programme"].upper()} STACK ALIGNMENT:
  Overall score  : {al["score"]}%  \
(core coverage {al["core_coverage"]}% × 60%  +  relevance {al["relevance"]}% × 40%)
  Core languages : matched [{matched_str}]
  Core missing   : [{missing_str}]
  Note: relevance = % of YOUR languages that are {a["programme"]}-relevant

TOP {len(a["top_repos"])} REPOSITORIES (ranked by stars + recency):
{chr(10).join(repo_lines)}

DETECTED GAPS:
{gaps_str}

═══════════════════════════════════════════════════════════════════
TASK: Write a professional, honest career advisory report for this {a["programme"]} \
{"organization" if a.get("is_org") else "student"}.

Your report MUST:
- Reference specific repo names, exact numbers, and metrics from above
- Explain WHY each recommendation matters from a hiring manager's perspective
  (e.g., "XYZ matters because companies like Stripe and Cloudflare hire for it heavily")
- Include industry context: what technologies are hot right now, what employers scan for
- Be direct and candid — if the portfolio is weak in an area, say so clearly
- Prioritize by impact on employability, not by what sounds nice

Respond in EXACTLY this structure (use markdown headers):

## Portfolio Verdict
3-4 sentences: overall impression, market competitiveness, and a single clear headline \
verdict (e.g., "Junior-ready for backend roles but not yet ML-competitive").

## Strengths to Leverage
2-3 genuine strengths worth highlighting in a CV/cover letter. For each: what it signals \
to a recruiter and which specific companies or roles value it most.

## Critical Gaps (Fix in Priority Order)
The top 3-4 issues that would cause a recruiter to pass. For each:
  - What the gap is (reference exact data)
  - Why it costs you interviews
  - The precise fix (specific repo name, exact action, time estimate)

## Projects to Build Next
2-3 project ideas tailored to {a["programme"]} internship expectations for someone at \
semester {a["semester"]}. For each:
  - Project name and 1-line pitch
  - Tech stack (be specific — library versions matter)
  - Why a hiring manager would be impressed (what skill gap it closes)
  - Realistic scope (weekend prototype vs 2-week project)

## 30-Day Battle Plan
A concrete week-by-week plan. Each week has 1-2 high-priority tasks with time estimates.

Rules:
- Mention actual repo names from the list above
- Use exact numbers from the metrics
- Tailor advice strictly to {a["programme"]} career paths and current industry demand
- Write as if you are personally accountable for getting this developer hired
"""
