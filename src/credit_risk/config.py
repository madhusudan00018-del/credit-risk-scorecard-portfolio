from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    random_state: int
    test_size: float
    validation_size: float


@dataclass(frozen=True)
class EconomicsConfig:
    lgd: float
    annual_margin_rate: float
    monthly_funding_cost: float
    false_negative_cost_rate: float
    false_positive_cost_rate: float


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read configuration files. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    with open(resolve_path(config_path), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_project_directories(config: dict[str, Any]) -> None:
    directories = [
        "data/raw",
        "data/interim",
        "data/processed",
        "models",
        "images",
    ]
    for path in directories:
        resolve_path(path).mkdir(parents=True, exist_ok=True)

    for path in config.get("paths", {}).values():
        resolved = resolve_path(path)
        if resolved.suffix:
            resolved.parent.mkdir(parents=True, exist_ok=True)


def project_config(config: dict[str, Any]) -> ProjectConfig:
    section = config["project"]
    return ProjectConfig(
        random_state=int(section.get("random_state", 42)),
        test_size=float(section.get("test_size", 0.20)),
        validation_size=float(section.get("validation_size", 0.20)),
    )


def economics_config(config: dict[str, Any]) -> EconomicsConfig:
    section = config["economics"]
    return EconomicsConfig(
        lgd=float(section.get("lgd", 0.45)),
        annual_margin_rate=float(section.get("annual_margin_rate", 0.18)),
        monthly_funding_cost=float(section.get("monthly_funding_cost", 0.006)),
        false_negative_cost_rate=float(section.get("false_negative_cost_rate", 0.45)),
        false_positive_cost_rate=float(section.get("false_positive_cost_rate", 0.025)),
    )
