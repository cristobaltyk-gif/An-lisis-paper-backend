import json
from pathlib import Path

DATA_DIR = Path("/tmp/screener")
DATA_DIR.mkdir(exist_ok=True)


def load_read(stream: str) -> set:
    path = DATA_DIR / f"{stream}_read.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def mark_as_read(stream: str, pmid: str):
    read = load_read(stream)
    read.add(pmid)
    path = DATA_DIR / f"{stream}_read.json"
    path.write_text(json.dumps(list(read)))


def is_read(stream: str, pmid: str) -> bool:
    return pmid in load_read(stream)


def get_all_read(stream: str) -> list:
    return list(load_read(stream))


def clear_read(stream: str):
    path = DATA_DIR / f"{stream}_read.json"
    if path.exists():
        path.write_text(json.dumps([]))
      
