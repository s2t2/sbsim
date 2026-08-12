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
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_strict_library(
    name = "utils",
    srcs = ["__init__.py"],
    visibility = ["//smart_control/reinforcement_learning:__pkg__"],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "//smart_control/reward:electricity_energy_cost",
        "//smart_control/reward:natural_gas_energy_cost",
        "//smart_control/reward:setpoint_energy_carbon_regret",
        "//smart_control/simulator:air_handler",
        "//smart_control/simulator:boiler",
        "//smart_control/simulator:building",
        "//smart_control/simulator:hvac_floorplan_based",
        "//smart_control/simulator:randomized_arrival_departure_occupancy",
        "//smart_control/simulator:simulator_building",
        "//smart_control/simulator:stochastic_convection_simulator",
        "//smart_control/simulator:tf_simulator",
        "//smart_control/simulator:weather_controller",
        "//smart_control/utils:constants", # This is smart_control.utils.constants
        "//smart_control/utils:controller_reader",
        "//smart_control/utils:controller_writer",
        "//smart_control/utils:environment_utils",
        "//smart_control/utils:histogram_reducer",
        "//smart_control/utils:observation_normalizer",
        "//third_party/py/gin",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        "//smart_control/utils:constants", # This is smart_control.utils.constants
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "data_processing",
    srcs = ["data_processing.py"],
    deps = [
        ":constants", # Refers to sm_rl.utils.constants
        "//smart_control/utils:controller_reader",
        "//smart_control/utils:conversion_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "data_processing_test",
    srcs = ["data_processing_test.py"],
    deps = [
        ":data_processing",
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "environment",
    srcs = ["environment.py"],
    deps = [
        ":constants", # Refers to sm_rl.utils.constants
        "//smart_control/environment",
        "//third_party/py/gin",
    ],
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        ":constants", # Refers to sm_rl.utils.constants
        "//third_party/py/numpy",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "time_utils",
    srcs = ["time_utils.py"],
    deps = [
        "//third_party/py/numpy",
    ],
)
