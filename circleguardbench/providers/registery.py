from typing import Dict, Type

import jinja2

from circleguardbench.models_config import InferenceEngine, ModelConfig, ModelType
from circleguardbench.providers.model_providers import BaseModelProvider


class ModelProvidersRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseModelProvider]] = {}

    def register(self, model_type: str):
        def decorator(cls: Type[BaseModelProvider]):
            self._registry[model_type] = cls
            return cls

        return decorator

    def create_provider(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ) -> BaseModelProvider:
        provider_type = self._get_provider_type(
            model_config.type, model_config.inference_engine
        )
        if provider_type not in self._registry:
            raise ValueError(f"Provider type '{provider_type}' is not registered")

        provider_class = self._registry[provider_type]
        return provider_class(model_config, prompts_templates)

    def _get_provider_type(
        self, model_type: ModelType, inference_engine: InferenceEngine
    ) -> str:
        if (
            model_type in [ModelType.LLM_SO, ModelType.OPENAI_MODERATION]
            and inference_engine != InferenceEngine.OPENAI_API
        ):
            raise ValueError(
                f"Model type {model_type.value} supports only 'openai_api' inference engine!"
            )

        if (
            model_type == ModelType.CLASSIFIER
            and inference_engine != InferenceEngine.TRANSFORMERS
        ) or (
            model_type != ModelType.CLASSIFIER
            and inference_engine == InferenceEngine.TRANSFORMERS
        ):
            raise ValueError(
                f"Model type {model_type.value} supports only 'transformers' inference engine for now!"
            )

        if model_type == ModelType.CLASSIFIER:
            return "transformers_clf"

        if inference_engine == InferenceEngine.OPENAI_API:
            if model_type == ModelType.LLM_SO:
                return "openai_structured"
            elif model_type in [ModelType.LLM_REGEXP, ModelType.ATLA_SELENE, ModelType.LLAMA_GUARD]:
                return "openai_regexp"
            else:
                return "openai_moderation"

        elif inference_engine == InferenceEngine.VLLM:
            return "vllm_regexp"
        elif inference_engine == InferenceEngine.SGLANG:
            return "sglang_regexp"


MODEL_PROVIDERS_REGISTRY = ModelProvidersRegistry()
