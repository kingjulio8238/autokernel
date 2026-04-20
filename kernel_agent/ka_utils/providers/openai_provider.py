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

"""OpenAI provider implementation."""

from .openai_base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider."""

    def __init__(self):
        super().__init__(api_key_env="OPENAI_API_KEY")

    @property
    def name(self) -> str:
        return "openai"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for OpenAI models."""
        if model_name.startswith("gpt-5"):
            return 32000
        elif model_name.startswith("gpt-4o"):
            return 16384  # GPT-4o max completion tokens
        elif model_name.startswith(("o3", "o4", "o1")):
            return 16384
        elif model_name.startswith("gpt-3.5"):
            return 4096
        else:
            return 8192
