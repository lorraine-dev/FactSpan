"""Publish a new version of the FactSpan Zenodo record with specific files.

Unlike Zenodo's default GitHub integration (which archives the entire repo
as a new, unrelated concept DOI), this publishes only the given files as a
new version of an existing concept, identified by any prior deposition id
in that concept's version chain.
"""
import argparse
import os
import sys

import requests

ZENODO_API = "https://zenodo.org/api"


def publish(api_base, token, record_id, version, description, publication_date, files, dry_run):
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    version = version.lstrip("vV")
    pub_date = publication_date[:10]

    print(f"Creating new version draft from record {record_id}...")
    r = session.post(f"{api_base}/deposit/depositions/{record_id}/actions/newversion")
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
        print(f"Uploading {filename}...")
        with open(path, "rb") as fh:
            ur = session.put(f"{bucket_url}/{filename}", data=fh)
            ur.raise_for_status()

    metadata = draft["metadata"]
    metadata.update(
        {
            "version": version,
            "description": description or metadata.get("description"),
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
        "--sandbox",
        action="store_true",
        help="Use sandbox.zenodo.org instead of production zenodo.org",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the new version draft but stop before publishing",
    )
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN environment variable is required")

    api_base = "https://sandbox.zenodo.org/api" if args.sandbox else ZENODO_API

    publish(
        api_base=api_base,
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
