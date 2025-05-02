import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from circleguardbench.models_config import ModelType
from circleguardbench.storage.objects import (
    GuardBenchMetrics,
    Leaderboard,
    LeaderboardEntry,
    Metrics,
)
from circleguardbench.storage.results import get_results_dir


def get_leaderboard_path(bench_name: str, base_dir: str = "results") -> Path:
    return get_results_dir(bench_name, base_dir) / "leaderboard.json"


def save_leaderboard(
    bench_name: str, leaderboard: Leaderboard, base_dir: str = "results"
) -> None:
    path = get_leaderboard_path(bench_name, base_dir)
    with open(path, "w", encoding="utf-8") as f:
        f.write(leaderboard.model_dump_json(indent=2))


def load_leaderboard(
    bench_name: str, base_dir: str = "results"
) -> Optional[Leaderboard]:
    path = get_leaderboard_path(bench_name, base_dir)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return Leaderboard.model_validate(data)


def create_empty_leaderboard() -> Leaderboard:
    return Leaderboard(entries=[], last_updated=datetime.now())


def update_leaderboard(
    bench_name: str,
    model_name: str,
    model_type: ModelType,
    metrics: Dict[str, GuardBenchMetrics],
    base_dir: str = "results",
) -> Leaderboard:
    leaderboard = load_leaderboard(bench_name, base_dir)
    if leaderboard is None:
        leaderboard = create_empty_leaderboard()

    avg_metrics = _calculate_avg_metrics(metrics)
    micro_avg_error_ratio, micro_avg_runtime_ms = _calculate_micro_avg_metrics(metrics)

    entry = LeaderboardEntry(
        model_name=model_name,
        model_type=model_type,
        per_category_metrics=metrics,
        avg_metrics=avg_metrics,
        micro_avg_error_ratio=micro_avg_error_ratio,
        micro_avg_runtime_ms=micro_avg_runtime_ms,
    )

    existing_entry = leaderboard.find_by_model_name(model_name)
    if existing_entry:
        for i, e in enumerate(leaderboard.entries):
            if e.model_name == model_name:
                leaderboard.entries[i] = entry
                break
    else:
        leaderboard.entries.append(entry)

    leaderboard.last_updated = datetime.now()

    save_leaderboard(bench_name, leaderboard, base_dir)

    return leaderboard


def _calculate_avg_metrics(
    metrics: Dict[str, GuardBenchMetrics],
) -> GuardBenchMetrics:
    """
    Calculates weighted average metrics across all categories.
    For f1, precision and recall, uses weighted average based on sample_count.
    For error_ratio and avg_runtime_ms, uses micro-average.

    Args:
        metrics: Dictionary with metrics by category

    Returns:
        Weighted average metrics
    """
    # Prepare lists for metrics and their weights
    default_prompts_metrics = []
    jailbreaked_prompts_metrics = []
    default_answers_metrics = []
    jailbreaked_answers_metrics = []

    # Collect metrics and determine weights based on sample_count
    for _, category_metrics in metrics.items():
        if category_metrics.default_prompts:
            default_prompts_metrics.append(
                (
                    category_metrics.default_prompts,
                    category_metrics.default_prompts.sample_count,
                )
            )
        if category_metrics.jailbreaked_prompts:
            jailbreaked_prompts_metrics.append(
                (
                    category_metrics.jailbreaked_prompts,
                    category_metrics.jailbreaked_prompts.sample_count,
                )
            )
        if category_metrics.default_answers:
            default_answers_metrics.append(
                (
                    category_metrics.default_answers,
                    category_metrics.default_answers.sample_count,
                )
            )
        if category_metrics.jailbreaked_answers:
            jailbreaked_answers_metrics.append(
                (
                    category_metrics.jailbreaked_answers,
                    category_metrics.jailbreaked_answers.sample_count,
                )
            )

    # Function to calculate weighted average metrics with micro-averages for error and runtime
    def avg_metrics(
        metrics_with_weights: List[tuple[Metrics, int]],
    ) -> Optional[Metrics]:
        if not metrics_with_weights:
            return None

        total_samples = sum(weight for _, weight in metrics_with_weights)

        if total_samples == 0:
            return None

        # Weighted average for f1, precision, recall and accuracy based on sample_count
        f1_binary = (
            sum(m.f1_binary * w for m, w in metrics_with_weights) / total_samples
        )
        recall_binary = (
            sum(m.recall_binary * w for m, w in metrics_with_weights) / total_samples
        )
        precision_binary = (
            sum(m.precision_binary * w for m, w in metrics_with_weights) / total_samples
        )
        accuracy = sum(m.accuracy * w for m, w in metrics_with_weights) / total_samples

        # Micro-average for error_ratio and avg_runtime_ms
        total_errors = sum(
            m.error_ratio * m.sample_count for m, _ in metrics_with_weights
        )
        error_ratio = total_errors / total_samples

        # For runtime, consider total execution time divided by total number of examples
        total_runtime = sum(
            m.avg_runtime_ms * m.sample_count for m, _ in metrics_with_weights
        )
        avg_runtime_ms = total_runtime / total_samples

        return Metrics(
            f1_binary=f1_binary,
            recall_binary=recall_binary,
            precision_binary=precision_binary,
            accuracy=accuracy,
            error_ratio=error_ratio,
            avg_runtime_ms=avg_runtime_ms,
            sample_count=total_samples,
        )

    return GuardBenchMetrics(
        default_prompts=avg_metrics(default_prompts_metrics),
        jailbreaked_prompts=avg_metrics(jailbreaked_prompts_metrics),
        default_answers=avg_metrics(default_answers_metrics),
        jailbreaked_answers=avg_metrics(jailbreaked_answers_metrics),
    )


def _calculate_micro_avg_metrics(
    metrics: Dict[str, GuardBenchMetrics],
) -> tuple[float, float]:
    """
    Calculates micro-average metrics across all categories.

    Args:
        metrics: Dictionary with metrics by category

    Returns:
        Tuple (micro_avg_error_ratio, micro_avg_runtime_ms)
    """
    # Here should be the logic for calculating micro-average metrics
    # This is a simplified implementation

    all_error_ratios = []
    all_runtimes = []

    for category_metrics in metrics.values():
        for metrics_obj in [
            category_metrics.default_prompts,
            category_metrics.jailbreaked_prompts,
            category_metrics.default_answers,
            category_metrics.jailbreaked_answers,
        ]:
            if metrics_obj:
                all_error_ratios.append(metrics_obj.error_ratio)
                all_runtimes.append(metrics_obj.avg_runtime_ms)

    if not all_error_ratios:
        return 0.0, 0.0

    return sum(all_error_ratios) / len(all_error_ratios), sum(all_runtimes) / len(
        all_runtimes
    )
