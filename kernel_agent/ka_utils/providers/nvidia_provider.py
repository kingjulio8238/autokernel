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

"""NVIDIA NIM provider (OpenAI-compatible, free-tier 40 RPM).

Endpoint: https://integrate.api.nvidia.com/v1
API key: NVIDIA_API_KEY (prefix `nvapi-...`)
"""

from .openai_base import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    """NVIDIA NIM API provider.

    Thin subclass of OpenAICompatibleProvider that points at NIM's
    OpenAI-compatible endpoint. Rate-limit handling lives in
    OpenAICompatibleProvider (task #2) and reads ``rpm_limit``.
    """

    # NIM free tier: 40 RPM. Consumed by rate-limiter (task #2).
    rpm_limit = 40

    def __init__(self):
        super().__init__(
            api_key_env="NVIDIA_API_KEY",
            base_url="https://integrate.api.nvidia.com/v1",
        )

    @property
    def name(self) -> str:
        return "nvidia"
