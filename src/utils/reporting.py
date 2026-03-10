from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _summary_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].fillna("unknown").value_counts().reset_index()
    counts.columns = ["value", "count"]
    counts.insert(0, "metric", column)
    return counts


def finalize_report(events: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    events_path = output_dir / "events.csv"
    summary_path = output_dir / "summary.csv"

    if not events:
        pd.DataFrame().to_csv(events_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        return {"events": events_path, "summary": summary_path}

    df = pd.DataFrame(events)
    df.to_csv(events_path, index=False)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour.fillna(-1).astype(int)

    summary_frames = []
    for column in ["gender", "age", "product", "hour"]:
        if column in df.columns:
            summary_frames.append(_summary_frame(df, column))

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary.to_csv(summary_path, index=False)

    return {"events": events_path, "summary": summary_path}
