load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "sac_networks",
    srcs = ["agents/networks/sac_networks.py"],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "sac_agent",
    srcs = ["agents/sac_agent.py"],
    deps = [
        ":sac_networks",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "base_observer",
    srcs = ["observers/base_observer.py"],
    deps = ["//third_party/py/tf_agents"],
)

pytype_strict_library(
    name = "composite_observer",
    srcs = ["observers/composite_observer.py"],
    deps = [
        ":base_observer",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "print_status_observer",
    srcs = ["observers/print_status_observer.py"],
    deps = [
        ":base_observer",
        ":constants",
        "//third_party/py/pandas",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "rendering_observer",
    srcs = ["observers/rendering_observer.py"],
    deps = [
        ":base_observer",
        ":config",
        ":constants",
        ":data_processing",
        "//smart_control/environment",
        "//smart_control/utils:building_renderer",
        "//third_party/py/matplotlib",
        "//third_party/py/pandas",
        "//third_party/py/pytz",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "schedule_policy",
    srcs = ["policies/schedule_policy.py"],
    deps = [
        ":constants",
        ":time_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "replay_buffer",
    srcs = ["replay_buffer/replay_buffer.py"],
    deps = [
        "//third_party/py/reverb",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "populate_starter_buffer",
    srcs = ["scripts/populate_starter_buffer.py"],
    deps = [
        ":composite_observer",
        ":config",
        ":environment",
        ":print_status_observer",
        ":replay_buffer",
        ":schedule_policy",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "train",
    srcs = ["scripts/train.py"],
    deps = [
        ":composite_observer",
        ":config",
        ":environment",
        ":print_status_observer",
        ":replay_buffer",
        ":sac_agent",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "config",
    srcs = ["utils/config.py"],
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
        "//smart_control/utils:constants",
        "//smart_control/utils:controller_reader",
        "//smart_control/utils:controller_writer",
        "//smart_control/utils:environment_utils",
        "//smart_control/utils:histogram_reducer",
        "//smart_control/utils:observation_normalizer",
        "//third_party/py/gin",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "config_test",
    srcs = ["utils/config_test.py"],
    deps = [
        "//smart_control/utils:constants",
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["utils/constants.py"],
)

pytype_strict_library(
    name = "data_processing",
    srcs = ["utils/data_processing.py"],
    deps = [
        ":constants",
        "//smart_control/utils:controller_reader",
        "//smart_control/utils:conversion_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "data_processing_test",
    srcs = ["utils/data_processing_test.py"],
    deps = [
        ":data_processing",
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "environment",
    srcs = ["utils/environment.py"],
    deps = [
        ":constants",
        "//smart_control/environment",
        "//third_party/py/gin",
    ],
)

pytype_strict_library(
    name = "metrics",
    srcs = ["utils/metrics.py"],
    deps = [
        ":constants",
        "//third_party/py/numpy",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "time_utils",
    srcs = ["utils/time_utils.py"],
    deps = ["//third_party/py/numpy"],
)

# __init__.py files
pytype_strict_library(
    name = "reinforcement_learning_init",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "agents_init",
    srcs = ["agents/__init__.py"],
)

pytype_strict_library(
    name = "networks_init",
    srcs = ["agents/networks/__init__.py"],
)

pytype_strict_library(
    name = "observers_init",
    srcs = ["observers/__init__.py"],
)

pytype_strict_library(
    name = "policies_init",
    srcs = ["policies/__init__.py"],
)

pytype_strict_library(
    name = "replay_buffer_init",
    srcs = ["replay_buffer/__init__.py"],
)

pytype_strict_library(
    name = "scripts_init",
    srcs = ["scripts/__init__.py"],
)

pytype_strict_library(
    name = "utils_init",
    srcs = ["utils/__init__.py"],
)
