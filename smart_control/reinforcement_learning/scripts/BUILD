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
load("//devtools/python/blaze:strict.bzl", "py_strict_binary")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_strict_library(
    name = "scripts_lib", # Renamed to avoid conflict
    srcs = ["__init__.py"],
    visibility = ["//smart_control/reinforcement_learning:__pkg__"],
)

# Library target for populate_starter_buffer.py
pytype_strict_library(
    name = "populate_starter_buffer_lib",
    srcs = ["populate_starter_buffer.py"],
    deps = [
        "//smart_control/reinforcement_learning/observers:composite_observer",
        "//smart_control/reinforcement_learning/observers:print_status_observer",
        "//smart_control/reinforcement_learning/policies:schedule_policy",
        "//smart_control/reinforcement_learning/replay_buffer",
        "//smart_control/reinforcement_learning/utils:config",
        "//smart_control/reinforcement_learning/utils:environment",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

# Binary target for populate_starter_buffer.py
py_strict_binary(
    name = "populate_starter_buffer",
    srcs = ["populate_starter_buffer.py"],
    main = "populate_starter_buffer.py", # Specifies the main entry point
    deps = [":populate_starter_buffer_lib"],
)

# Library target for train.py
pytype_strict_library(
    name = "train_lib",
    srcs = ["train.py"],
    deps = [
        "//smart_control/reinforcement_learning/agents:sac_agent",
        "//smart_control/reinforcement_learning/observers:composite_observer",
        "//smart_control/reinforcement_learning/observers:print_status_observer",
        "//smart_control/reinforcement_learning/replay_buffer",
        "//smart_control/reinforcement_learning/utils:config",
        "//smart_control/reinforcement_learning/utils:environment",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

# Binary target for train.py
py_strict_binary(
    name = "train",
    srcs = ["train.py"],
    main = "train.py", # Specifies the main entry point
    deps = [":train_lib"],
)
