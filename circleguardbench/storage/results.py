import json
import logging
from pathlib import Path
from typing import List, Optional, Iterator
from circleguardbench.storage.objects import BenchResult

logger = logging.getLogger(__name__)


def get_results_dir(bench_name: str, base_dir: str = "results") -> Path:
    """Gets and creates a directory for the results of a specific benchmark"""
    results_dir = Path(base_dir) / bench_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_model_file_path(
    bench_name: str, model_name: str, base_dir: str = "results"
) -> Path:
    """Gets the path to the results file for a specific model"""
    safe_name = model_name.replace("/", "_")
    return get_results_dir(bench_name, base_dir) / f"{safe_name}.jsonl"


def append_result(
    bench_name: str, model_name: str, result: BenchResult, base_dir: str = "results"
) -> None:
    """Adds a new result to the model file"""
    file_path = get_model_file_path(bench_name, model_name, base_dir)
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Error appending result for model {model_name}: {e}")
        raise


def get_results(
    bench_name: str, model_name: str, base_dir: str = "results"
) -> Iterator[BenchResult]:
    """Reads all results for a model"""
    file_path = get_model_file_path(bench_name, model_name, base_dir)
    if not file_path.exists():
        return iter([])

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield BenchResult.from_json(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading results for model {model_name}: {e}")
        raise


def clear_results(bench_name: str, model_name: str, base_dir: str = "results") -> None:
    """Clears the results file for the specified model"""
    file_path = get_model_file_path(bench_name, model_name, base_dir)
    if file_path.exists():
        try:
            # Open file in write mode to truncate its contents
            open(file_path, "w", encoding="utf-8").close()
            logger.info(f"Results for model {model_name} have been cleared")
        except Exception as e:
            logger.error(f"Error clearing results for model {model_name}: {e}")
            raise


def count_results(bench_name: str, model_name: str, base_dir: str = "results") -> int:
    """Counts the number of results for a model"""
    file_path = get_model_file_path(bench_name, model_name, base_dir)
    if not file_path.exists():
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        logger.error(f"Error counting results for model {model_name}: {e}")
        raise


def get_available_models(bench_name: str, base_dir: str = "results") -> List[str]:
    """Returns a list of models for which results are available"""
    results_dir = get_results_dir(bench_name, base_dir)
    models = []
    for file_path in results_dir.glob("*.jsonl"):
        model_name = file_path.stem.replace("_", "/")
        models.append(model_name)
    return models


def get_results_by_hash(
    bench_name: str, model_name: str, prompt_hash: str, base_dir: str = "results"
) -> Optional[BenchResult]:
    """Searches for a result by prompt hash"""
    for result in get_results(bench_name, model_name, base_dir):
        if result.prompt_hash == prompt_hash:
            return result
    return None
