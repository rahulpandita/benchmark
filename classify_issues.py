#!/usr/bin/env python3
"""
classify_issues.py — Phase 1: Broad candidate downloader for GitHub issues.

This is a two-phase approach to building a benchmark dataset of closed issues:

  Phase 1 (this script):
    Download closed issues from a GitHub repository using the GitHub search
    API and bucket them loosely into three categories based on closure reason
    and contributor status.  Rich metadata is saved for each candidate so
    that Phase 2 can be done offline.

  Phase 2 (manual review):
    A human reviewer goes through each candidate holistically and assigns a
    final classification:
      1. New contributor slop (AI-generated / low-quality)
      2. New contributor not-planned (legitimate but won't fix)
      3. New contributor worked-upon (issue was actually addressed)
      4. Regular contributor slop

Buckets produced by this script:
  A — "not_planned" issues filed by new contributors  (candidates for 1 & 2)
  B — "completed"   issues filed by new contributors  (candidates for 3)
  C — "not_planned" issues filed by regular contributors (candidates for 4)

Definitions:
  - New contributor:     ≤3 prior issues+PRs in the repo at time of filing
                         (the issue itself is excluded from the count).
  - Regular contributor: >3 prior issues+PRs at time of filing.

Usage:
  python3 classify_issues.py --repo owner/repo [--target 20] [--output FILE]

Requires:
  - ``gh`` CLI authenticated with GitHub
  - Python 3.8+
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Contributor count cache — keyed by "repo:username:YYYY-MM-DD"
# ---------------------------------------------------------------------------
_contributor_cache: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# GitHub API helper
# ---------------------------------------------------------------------------

def gh_api(endpoint: str) -> Any:
    """Call the GitHub API via ``gh api`` and return parsed JSON.

    Returns an empty dict on failure so callers can safely use ``.get()``.
    """
    cmd = [
        "gh", "api", endpoint,
        "--header", "Accept: application/vnd.github+json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(
            f"  [WARN] gh api failed for {endpoint}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return {}
    text = result.stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Issue fetching via search API
# ---------------------------------------------------------------------------

def search_issues(
    repo: str,
    reason: str,
    max_pages: int = 5,
    per_page: int = 100,
    exclude_labels: Optional[List[str]] = None,
) -> List[dict]:
    """Fetch closed issues for a given ``reason`` using the search API.

    Args:
        repo: GitHub repository in ``owner/repo`` format.
        reason: ``"not_planned"`` or ``"completed"``.
        max_pages: Maximum number of pages to fetch.
        per_page: Results per page (max 100 for search API).
        exclude_labels: Labels to exclude from search results.

    Returns:
        A list of issue dicts (PRs are filtered out).
    """
    all_issues: List[dict] = []
    for page in range(1, max_pages + 1):
        q = f"repo:{repo}+is:issue+is:closed+reason:{reason}"
        for label in (exclude_labels or []):
            q += f"+-label:{label}"
        endpoint = (
            f"/search/issues?q={q}"
            f"&sort=created&order=desc&per_page={per_page}&page={page}"
        )
        print(f"    page {page}/{max_pages} ...", end=" ", flush=True)
        data = gh_api(endpoint)
        items = data.get("items", [])
        total = data.get("total_count", "?")

        # Filter out PRs (search results can include them)
        issues = [i for i in items if "pull_request" not in i]
        all_issues.extend(issues)
        print(f"got {len(issues)} issues (total available: {total})")

        if len(items) < per_page:
            break
        time.sleep(3)  # search API is more rate-limited

    return all_issues


# ---------------------------------------------------------------------------
# Contributor classification
# ---------------------------------------------------------------------------

def get_contributor_count(repo: str, username: str, before_date: str) -> int:
    """Return the number of prior issues+PRs by *username* in the repo.

    The count is date-filtered (``created<=YYYY-MM-DD``) and then reduced by
    1 to exclude the issue itself (which is included in the search results).
    """
    date_str = before_date[:10]  # YYYY-MM-DD
    cache_key = f"{repo}:{username}:{date_str}"
    if cache_key in _contributor_cache:
        return _contributor_cache[cache_key]

    endpoint = (
        f"/search/issues?q=repo:{repo}+author:{username}"
        f"+created:<={date_str}&per_page=1"
    )
    data = gh_api(endpoint)
    raw_count = data.get("total_count", 0) if isinstance(data, dict) else 0
    count = max(0, raw_count - 1)  # subtract 1 to exclude the current issue
    _contributor_cache[cache_key] = count
    time.sleep(2)  # respect search API rate limit (30 req/min)
    return count


def is_new_contributor(repo: str, username: str, before_date: str, threshold: int = 3) -> bool:
    """A new contributor has ≤ *threshold* prior contributions at filing time."""
    return get_contributor_count(repo, username, before_date) <= threshold


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def days_open(issue: dict) -> float:
    """Calculate how many days an issue was open."""
    created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
    closed_raw = issue.get("closed_at")
    if not closed_raw:
        return -1.0
    closed = datetime.fromisoformat(closed_raw.replace("Z", "+00:00"))
    return round((closed - created).total_seconds() / 86400, 2)


def format_issue(issue: dict, contrib_count: int, is_new: bool) -> dict:
    """Extract rich metadata from an issue for the output file."""
    body = issue.get("body") or ""
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body_preview": body[:500],
        "url": issue["html_url"],
        "author": issue["user"]["login"],
        "author_contrib_count": contrib_count,
        "is_new_contributor": is_new,
        "labels": [l["name"] for l in issue.get("labels", [])],
        "created_at": issue["created_at"],
        "closed_at": issue.get("closed_at", ""),
        "days_open": days_open(issue),
        "comments": issue.get("comments", 0),
        "state_reason": issue.get("state_reason", ""),
    }


# ---------------------------------------------------------------------------
# Bucket filling
# ---------------------------------------------------------------------------

def fill_bucket(
    repo: str,
    issues: List[dict],
    bucket_name: str,
    want_new: bool,
    target: int,
) -> List[dict]:
    """Classify contributors and collect up to *target* candidates.

    Args:
        repo: GitHub repository in ``owner/repo`` format.
        issues: Raw issue dicts from the search API.
        bucket_name: Human-readable name (for progress messages).
        want_new: ``True`` to keep new contributors, ``False`` for regulars.
        target: Stop after collecting this many candidates.

    Returns:
        A list of formatted candidate dicts.
    """
    candidates: List[dict] = []
    for issue in issues:
        if len(candidates) >= target:
            break

        author = issue["user"]["login"]
        if author.endswith("[bot]"):
            continue

        created_at = issue["created_at"]
        count = get_contributor_count(repo, author, created_at)
        new = count <= 3

        if new != want_new:
            continue

        candidates.append(format_issue(issue, count, new))
        label = "new" if new else "regular"
        print(
            f"    [{len(candidates)}/{target}] #{issue['number']} "
            f"by {author} ({count} prior contribs, {label})"
        )

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Download broad candidate issues into buckets "
        "for later manual review.",
    )
    parser.add_argument(
        "--repo", type=str, required=True,
        help="GitHub repository in owner/repo format (e.g. tldraw/tldraw)",
    )
    parser.add_argument(
        "--target", type=int, default=20,
        help="Number of candidates to collect per bucket (default: 20)",
    )
    parser.add_argument(
        "--pages", type=int, default=5,
        help="Number of search result pages to fetch per bucket (default: 5, max 10)",
    )
    parser.add_argument(
        "--exclude-labels", type=str, default=None,
        help="Comma-separated labels to exclude (e.g. 'automation,bot,automated-analysis')",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: candidates_<owner>_<repo>.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.repo.replace("/", "_")
        args.output = f"candidates_{safe_name}.json"

    repo = args.repo
    target = args.target
    exclude_labels = [l.strip() for l in args.exclude_labels.split(",")] if args.exclude_labels else None
    print(f"=== Phase 1: Downloading candidate issues from {repo} ===")
    print(f"    Target: {target} candidates per bucket")
    if exclude_labels:
        print(f"    Excluding labels: {exclude_labels}")
    print()

    # ------------------------------------------------------------------
    # Step 1 — Fetch not_planned issues
    # ------------------------------------------------------------------
    print("[1/4] Fetching not_planned issues via search API ...")
    not_planned_issues = search_issues(repo, "not_planned", max_pages=args.pages, exclude_labels=exclude_labels)
    print(f"  → {len(not_planned_issues)} not_planned issues fetched\n")

    # ------------------------------------------------------------------
    # Step 2 — Fetch completed issues
    # ------------------------------------------------------------------
    print("[2/4] Fetching completed issues via search API ...")
    completed_issues = search_issues(repo, "completed", max_pages=args.pages, exclude_labels=exclude_labels)
    print(f"  → {len(completed_issues)} completed issues fetched\n")

    # ------------------------------------------------------------------
    # Step 3 — Fill buckets (requires per-author API calls)
    # ------------------------------------------------------------------
    print("[3/4] Classifying contributors and filling buckets ...\n")

    print(f"  Bucket A — not_planned + new contributor (target: {target})")
    bucket_a = fill_bucket(repo, not_planned_issues, "A", want_new=True, target=target)
    print(f"  → Bucket A: {len(bucket_a)} candidates\n")

    print(f"  Bucket B — completed + new contributor (target: {target})")
    bucket_b = fill_bucket(repo, completed_issues, "B", want_new=True, target=target)
    print(f"  → Bucket B: {len(bucket_b)} candidates\n")

    print(f"  Bucket C — not_planned + regular contributor (target: {target})")
    bucket_c = fill_bucket(repo, not_planned_issues, "C", want_new=False, target=target)
    print(f"  → Bucket C: {len(bucket_c)} candidates\n")

    # ------------------------------------------------------------------
    # Step 4 — Write output
    # ------------------------------------------------------------------
    print("[4/4] Writing results ...")

    output = {
        "metadata": {
            "repository": repo,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "target_per_bucket": target,
            "not_planned_issues_fetched": len(not_planned_issues),
            "completed_issues_fetched": len(completed_issues),
            "contributor_threshold": 3,
            "description": (
                "Phase 1 broad candidate download. Bucket A and C come from "
                "not_planned issues; Bucket B from completed issues. "
                "Contributor status is determined by prior issues+PRs at "
                "time of filing (<=3 = new, >3 = regular). The issue itself "
                "is excluded from the count."
            ),
        },
        "bucket_A_not_planned_new_contributor": bucket_a,
        "bucket_B_completed_new_contributor": bucket_b,
        "bucket_C_not_planned_regular_contributor": bucket_c,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Bucket A (not_planned + new):     {len(bucket_a)}/{target}")
    print(f"  Bucket B (completed + new):       {len(bucket_b)}/{target}")
    print(f"  Bucket C (not_planned + regular): {len(bucket_c)}/{target}")
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
