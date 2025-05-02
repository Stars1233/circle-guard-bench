from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import json
from datetime import datetime

from circleguardbench.models_config import ModelType


class Verdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class ModelOutput(BaseModel):
    verdict: Verdict
    raw_prediction: Optional[str | float] = Field(
        default=None, description="Raw prediction output from model"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if any"
    )
    run_time_ms: Optional[float] = Field(
        default=None, description="Time of work in milliseconds"
    )


class BenchResult(BaseModel):
    prompt_hash: str
    result: ModelOutput

    def to_json(self) -> Dict[str, Any]:
        return json.loads(self.model_dump_json())

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "BenchResult":
        return cls.model_validate(data)


class Metrics(BaseModel):
    f1_binary: float
    recall_binary: float
    precision_binary: float
    accuracy: float
    error_ratio: float
    avg_runtime_ms: float
    sample_count: int


class GuardBenchMetrics(BaseModel):
    default_prompts: Optional[Metrics] = None
    jailbreaked_prompts: Optional[Metrics] = None
    default_answers: Optional[Metrics] = None
    jailbreaked_answers: Optional[Metrics] = None


class LeaderboardEntry(BaseModel):
    model_name: str
    model_type: ModelType
    per_category_metrics: Dict[str, GuardBenchMetrics]
    avg_metrics: GuardBenchMetrics
    micro_avg_error_ratio: float
    micro_avg_runtime_ms: float


class Leaderboard(BaseModel):
    entries: List[LeaderboardEntry] = Field(default_factory=list)
    last_updated: datetime

    def find_by_model_name(self, model_name: str) -> Optional[LeaderboardEntry]:
        for entry in self.entries:
            if entry.model_name == model_name:
                return entry
        return None
