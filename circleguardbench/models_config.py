import os
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional, Union, Any
from enum import Enum
from pathlib import Path
import json


# Enum for model types
class ModelType(str, Enum):
    WC_GUARD = "wc_guard"
    LLAMA_GUARD = "llama_guard"
    ATLA_SELENE = "atla_selene"
    LLM_REGEXP = "llm_regexp"
    LLM_SO = "llm_so"
    CLASSIFIER = "classifier"
    OPENAI_MODERATION = "openai_moderation"


# Enum for evaluation types
class EvalOn(str, Enum):
    PROMPTS = "prompts"
    ANSWERS = "answers"
    ALL = "all"


# Enum for inference engines
class InferenceEngine(str, Enum):
    OPENAI_API = "openai_api"  # for LLM_REGEXP, LLM_SO and OPENAI_MODERATION
    VLLM = "vllm"
    SGLANG = "sglang"
    TRANSFORMERS = "transformers"  # for transformers_clf


# OpenAI API parameters
class OpenAIParams(BaseModel):
    api_model_name: str = Field(..., description="Model name on server")
    endpoint: str = Field(..., description="API endpoint URL")
    api_key: Optional[str] = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        description="API key for authentication. Defaults to OPENAI_API_KEY environment variable.",
    )


# Model for model configuration (models.json)
class ModelConfig(BaseModel):
    name: str = Field(..., description="Unique name for the model")
    type: ModelType = Field(..., description="Type of safety model")
    eval_on: EvalOn = Field(
        default=EvalOn.ALL, description="What to evaluate: prompts, answers, or both"
    )
    inference_engine: InferenceEngine = Field(
        ..., description="Engine used for inference (openai_api or vllm)"
    )
    params: Optional[Union[OpenAIParams, Dict[str, Any]]] = Field(
        default=None, description="Engine-specific parameters. Required for OpenAI API."
    )
    max_concurrency: Optional[int] = Field(
        default=4, description="Max concurrency for parallel inference of this model"
    )
    use_cot: Optional[bool] = Field(
        default=False,
        description="Use CoT Reasonong before generating the safety verdict, applicable only to 'llm_regexp' and 'llm_so' model types",
    )

    @model_validator(mode="after")
    def validate_params(self) -> "ModelConfig":
        engine = self.inference_engine
        params = self.params

        if self.type == ModelType.LLM_SO and engine != InferenceEngine.OPENAI_API:
            raise ValueError(
                "LLM_SO models can only use OPENAI_API as inference engine"
            )

        if self.use_cot and self.type not in [ModelType.LLM_SO, ModelType.LLM_REGEXP]:
            raise ValueError(
                "CoT reasoning is only appplicable to 'llm_regexp' and 'llm_so' model types"
            )

        if engine == InferenceEngine.OPENAI_API:
            if not params:
                raise ValueError("OpenAI API requires params with 'endpoint'")
            if (
                not isinstance(params, dict)
                and not isinstance(params, OpenAIParams)
                or isinstance(params, dict)
                and "endpoint" not in params
            ):
                raise ValueError("OpenAI API requires params with 'endpoint'")
            # Convert dict to OpenAIParams if it's not already
            if not isinstance(params, OpenAIParams):
                self.params = OpenAIParams(**params)

        elif engine == InferenceEngine.VLLM:
            # For VLLM, params are optional but if provided must include 'model'
            if params and (not isinstance(params, dict) or "model" not in params):
                raise ValueError("VLLM params must include 'model' field")
            # Note: We don't convert to EngineArgs here, it will be done when needed

        elif engine == InferenceEngine.SGLANG:
            # For VLLM, params are optional but if provided must include 'model'
            if params and (not isinstance(params, dict) or "model_path" not in params):
                raise ValueError("SGLANG params must include 'model_path' field")
            # Note: We don't convert to EngineArgs here, it will be done when needed

        return self


class ModelsRegistry:
    """
    Registry for safety models that allows accessing models by name.
    """

    def __init__(self, models: List[ModelConfig]):
        self._models = {model.name: model for model in models}

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get a model by name"""
        return self._models.get(name)

    def __getitem__(self, name: str) -> ModelConfig:
        """Get a model by name, raises KeyError if not found"""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]

    def __contains__(self, name: str) -> bool:
        """Check if a model exists by name"""
        return name in self._models

    def list(self) -> List[str]:
        """List all model names"""
        return list(self._models.keys())

    def filter(
        self, model_type: Optional[ModelType] = None, eval_on: Optional[EvalOn] = None
    ) -> List[ModelConfig]:
        """Filter models by type and/or eval_on"""
        result = list(self._models.values())

        if model_type:
            result = [m for m in result if m.type == model_type]

        if eval_on:
            result = [
                m for m in result if m.eval_on == eval_on or m.eval_on == EvalOn.ALL
            ]

        return result

    def __len__(self) -> int:
        """Return the number of models in the registry"""
        return len(self._models)


def load_models_configs_registery(config_path: Path) -> ModelsRegistry:
    """
    Loads and validates the models configuration.
    Returns a ModelsRegistry for easy access to models by name.
    """
    with open(config_path, "r") as f:
        models_data = json.load(f)
    models = [ModelConfig(**model_data) for model_data in models_data]
    return ModelsRegistry(models)
