"""Publish a new version of the FactSpan Zenodo record with specific files.

Unlike Zenodo's default GitHub integration (which archives the entire repo
as a new, unrelated concept DOI), this publishes only the given files as a
new version of an existing concept, identified by any prior deposition id
in that concept's version chain.
"""
import argparse
import html
import os
import re
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ZENODO_API = "https://zenodo.org/api"
UPLOAD_RETRIES = 3
UPLOAD_RETRY_BACKOFF_SECONDS = 5


def make_session(token):
    """A session that retries transient 5xx errors on GET/POST/DELETE.

    PUT (file upload) is deliberately excluded: it streams a file object as
    the request body, and the upload loop below already retries those by
    reopening the file fresh, which a body-preserving low-level retry
    can't do safely.
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "DELETE"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _inline_html(text):
    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def changelog_to_html(text):
    """Convert our CHANGELOG's Markdown-ish structure to HTML.

    Zenodo's description field renders as HTML, not Markdown, so plain
    newlines collapse unless wrapped in real block tags. Soft-wrapped
    continuation lines (no blank line, no new header/bullet) are joined
    into the block they continue, matching Markdown's lazy-continuation
    convention.
    """
    parts = []
    in_list = False
    state = None  # 'li', 'p', or None
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            content = _inline_html(" ".join(buffer))
            tag = "li" if state == "li" else "p"
            parts.append(f"<{tag}>{content}</{tag}>")
            buffer = []

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            state = None
            continue

        header = re.match(r"^(#{1,6})\s+(.*)", line)
        bullet = re.match(r"^[-*]\s+(.*)", line)

        if header:
            flush()
            if in_list:
                parts.append("</ul>")
                in_list = False
            level = min(len(header.group(1)) + 1, 6)
            parts.append(f"<h{level}>{_inline_html(header.group(2))}</h{level}>")
            state = None
        elif bullet:
            flush()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            state = "li"
            buffer = [bullet.group(1)]
        elif state in ("li", "p"):
            buffer.append(line)
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            state = "p"
            buffer = [line]

    flush()
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


def find_existing_draft(session, api_base, record_id):
    """Find a pending unpublished draft for the same concept as record_id.

    The newversion action's own links.latest_draft is only reliable on the
    response of a *successful* newversion call. A plain GET on the original
    (published) deposition does not reliably expose the same link, so
    instead search the account's own draft depositions by conceptrecid.
    """
    r = session.get(f"{api_base}/deposit/depositions/{record_id}")
    r.raise_for_status()
    concept_id = r.json().get("conceptrecid")

    r = session.get(f"{api_base}/deposit/depositions", params={"status": "draft", "size": 100})
    r.raise_for_status()
    for dep in r.json():
        if dep.get("conceptrecid") == concept_id:
            return dep["links"]["self"]
    return None


def publish(api_base, token, record_id, version, description, publication_date, files, dry_run):
    session = make_session(token)

    version = version.lstrip("vV")
    pub_date = publication_date[:10]

    print(f"Creating new version draft from record {record_id}...")
    r = session.post(f"{api_base}/deposit/depositions/{record_id}/actions/newversion")
    if r.status_code == 400:
        print("A draft already exists for this record; searching for it...")
        draft_url = find_existing_draft(session, api_base, record_id)
        if draft_url is None:
            raise RuntimeError(
                "newversion returned 400 (draft already exists) but no matching "
                "draft was found among this account's drafts."
            )
    else:
        r.raise_for_status()
        draft_url = r.json()["links"]["latest_draft"]
    draft_id = draft_url.rstrip("/").split("/")[-1]
    print(f"Draft deposition id: {draft_id}")

    r = session.get(draft_url)
    r.raise_for_status()
    draft = r.json()

    for f in draft.get("files", []):
        print(f"Deleting inherited file: {f['filename']}")
        dr = session.delete(f"{api_base}/deposit/depositions/{draft_id}/files/{f['id']}")
        dr.raise_for_status()

    bucket_url = draft["links"]["bucket"]
    for path in files:
        filename = os.path.basename(path)
        for attempt in range(1, UPLOAD_RETRIES + 1):
            print(f"Uploading {filename} (attempt {attempt}/{UPLOAD_RETRIES})...")
            try:
                with open(path, "rb") as fh:
                    ur = session.put(f"{bucket_url}/{filename}", data=fh)
                    ur.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.SSLError) as exc:
                if attempt == UPLOAD_RETRIES:
                    raise
                print(f"Upload failed ({exc}); retrying in {UPLOAD_RETRY_BACKOFF_SECONDS}s...")
                time.sleep(UPLOAD_RETRY_BACKOFF_SECONDS)

    metadata = draft["metadata"]
    metadata.update(
        {
            "version": version,
            "description": changelog_to_html(description) if description else metadata.get("description"),
            "publication_date": pub_date,
        }
    )
    mr = session.put(f"{api_base}/deposit/depositions/{draft_id}", json={"metadata": metadata})
    mr.raise_for_status()

    if dry_run:
        print(f"Dry run: draft {draft_id} prepared but not published.")
        print(f"Review at: https://zenodo.org/deposit/{draft_id}")
        return

    pr = session.post(f"{api_base}/deposit/depositions/{draft_id}/actions/publish")
    pr.raise_for_status()
    published = pr.json()
    print(f"Published: {published.get('doi_url', published.get('doi'))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Files to upload (replaces all existing files)")
    parser.add_argument("--record-id", required=True, help="Any existing deposition id in the target concept")
    parser.add_argument("--version", required=True, help="Version string, e.g. v1.1.0 or 1.1.0")
    parser.add_argument("--description", required=True)
    parser.add_argument("--publication-date", required=True, help="ISO date/datetime; truncated to YYYY-MM-DD")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the new version draft but stop before publishing",
    )
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN environment variable is required")

    publish(
        api_base=ZENODO_API,
        token=token,
        record_id=args.record_id,
        version=args.version,
        description=args.description,
        publication_date=args.publication_date,
        files=args.files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
