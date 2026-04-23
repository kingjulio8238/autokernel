# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""External available models for KernelAgent."""

from kernel_agent.ka_utils.providers.model_config import ModelConfig
from kernel_agent.ka_utils.providers.openai_provider import OpenAIProvider
from kernel_agent.ka_utils.providers.anthropic_provider import AnthropicProvider
from kernel_agent.ka_utils.providers.relay_provider import RelayProvider
from kernel_agent.ka_utils.providers.ollama_provider import OllamaProvider
from kernel_agent.ka_utils.providers.nvidia_provider import NvidiaProvider


# Registry of all available models (external/OSS version)
AVAILABLE_MODELS = [
    # OpenAI GPT-4o Models
    ModelConfig(
        name="gpt-4o",
        provider_classes=[OpenAIProvider],
        description="GPT-4o - strong coding and kernel generation",
    ),
    ModelConfig(
        name="gpt-4o-mini",
        provider_classes=[OpenAIProvider],
        description="GPT-4o Mini - fast and cheap",
    ),
    ModelConfig(
        name="o3-mini",
        provider_classes=[OpenAIProvider],
        description="o3-mini - strong reasoning model",
    ),
    ModelConfig(
        name="o4-mini",
        provider_classes=[OpenAIProvider],
        description="OpenAI o4-mini - fast reasoning model",
    ),
    # OpenAI GPT-5 Model (Only GPT-5)
    ModelConfig(
        name="gpt-5",
        provider_classes=[RelayProvider, OpenAIProvider],
        description="GPT-5 flagship model (Released Aug 2025)",
    ),
    ModelConfig(
        name="gpt-5.2",
        provider_classes=[OpenAIProvider],
        description="GPT-5.2 flagship model (Released Dec 2025)",
    ),
    # Anthropic Claude 4 Models (Latest)
    ModelConfig(
        name="claude-opus-4-6",
        provider_classes=[AnthropicProvider],
        description="Claude 4.6 Opus - most intelligent (Released Feb 2026)",
    ),
    ModelConfig(
        name="claude-sonnet-4-6",
        provider_classes=[AnthropicProvider],
        description="Claude 4.6 Sonnet - fast and powerful (Released Feb 2026)",
    ),
    ModelConfig(
        name="claude-opus-4-1-20250805",
        provider_classes=[AnthropicProvider],
        description="Claude 4.1 Opus - most capable (Released Aug 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-20250514",
        provider_classes=[AnthropicProvider],
        description="Claude 4 Sonnet - high performance (Released May 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-5-20250929",
        provider_classes=[AnthropicProvider],
        description="Claude 4.5 Sonnet - latest balanced model (Released Sep 2025)",
    ),
    ModelConfig(
        name="gcp-claude-4-sonnet",
        provider_classes=[RelayProvider],
        description="[Relay] Claude 4 Sonnet",
    ),
    ModelConfig(
        name="claude-opus-4.5",
        provider_classes=[RelayProvider],
        description="Claude 4.5 Opus (Released Nov 2025)",
    ),
    ModelConfig(
        name="gpt-5-2",
        provider_classes=[RelayProvider],
        description="GPT-5.2 flagship model (Dec 2025) - Note the name is different from the OpenAI model",
    ),
    # Ollama / Self-hosted Models
    ModelConfig(
        name="KernelLLM",
        provider_classes=[OllamaProvider],
        description="facebook/KernelLLM - Llama 3.1 8B fine-tuned on 25K Triton pairs (20.2 pass@1 on KernelBench L1)",
    ),
    ModelConfig(
        name="ollama/KernelLLM",
        provider_classes=[OllamaProvider],
        description="facebook/KernelLLM via Ollama - Triton kernel specialist",
    ),
    ModelConfig(
        name="ollama/codellama",
        provider_classes=[OllamaProvider],
        description="CodeLlama via Ollama - general code generation",
    ),
    ModelConfig(
        name="ollama/deepseek-coder-v2",
        provider_classes=[OllamaProvider],
        description="DeepSeek Coder V2 via Ollama - strong coding model",
    ),
    # NVIDIA NIM free-tier models
    # Endpoint: https://integrate.api.nvidia.com/v1 (40 RPM free tier).
    # Default generator is deepseek-ai/deepseek-v3.2; critic role uses
    # moonshotai/kimi-k2-thinking (NOT the -instruct variant).
    ModelConfig(
        name="deepseek-ai/deepseek-v3.2",
        provider_classes=[NvidiaProvider],
        description="DeepSeek V3.2 - strongest open-source coder for Triton/CUDA kernels (94.67% KernelBench-Triton)",
    ),
    ModelConfig(
        name="moonshotai/kimi-k2-thinking",
        provider_classes=[NvidiaProvider],
        description="Kimi K2 Thinking - SOTA reasoning for critic/reflexion roles (84.5 GPQA, 256K ctx)",
    ),
    ModelConfig(
        name="moonshotai/kimi-k2-instruct",
        provider_classes=[NvidiaProvider],
        description="Kimi K2 Instruct - general-purpose, non-thinking variant",
    ),
    ModelConfig(
        name="minimaxai/minimax-m2.7",
        provider_classes=[NvidiaProvider],
        description="MiniMax M2.7 - 230B MoE, broad capability",
    ),
    ModelConfig(
        name="openai/gpt-oss-120b",
        provider_classes=[NvidiaProvider],
        description="GPT-OSS-120B - open-weights GPT-OSS, diversification option",
    ),
    ModelConfig(
        name="thudm/glm-5-air",
        provider_classes=[NvidiaProvider],
        description="GLM-5 Air - fast and cheap (Haiku-class)",
    ),
    ModelConfig(
        name="meta/llama-3.3-70b-instruct",
        provider_classes=[NvidiaProvider],
        description="Llama 3.3 70B Instruct - general-purpose baseline",
    ),
]
