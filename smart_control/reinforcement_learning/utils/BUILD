load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "utils_init",
    srcs = ["__init__.py"],
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
    srcs = ["config_test.py"],
    deps = [
        "//smart_control/utils:constants",
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
    name = "data_processing",
    srcs = ["data_processing.py"],
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
        ":constants",
        "//smart_control/environment",
        "//third_party/py/gin",
    ],
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        ":constants",
        "//third_party/py/numpy",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "time_utils",
    srcs = ["time_utils.py"],
    deps = ["//third_party/py/numpy"],
)
