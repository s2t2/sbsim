# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_strict_library(
    name = "agents",
    srcs = ["__init__.py"],
    visibility = ["//smart_control/reinforcement_learning:__pkg__"],
)

pytype_strict_library(
    name = "sac_agent",
    srcs = ["sac_agent.py"],
    deps = [
        "//smart_control/reinforcement_learning/agents/networks:sac_networks",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)
