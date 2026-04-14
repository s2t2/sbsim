load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "observers_init",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "base_observer",
    srcs = ["base_observer.py"],
    deps = ["//third_party/py/tf_agents"],
)

pytype_strict_library(
    name = "composite_observer",
    srcs = ["composite_observer.py"],
    deps = [
        ":base_observer",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "print_status_observer",
    srcs = ["print_status_observer.py"],
    deps = [
        ":base_observer",
        "//smart_control/reinforcement_learning/utils:constants",
        "//third_party/py/pandas",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "rendering_observer",
    srcs = ["rendering_observer.py"],
    deps = [
        ":base_observer",
        "//smart_control/environment",
        "//smart_control/reinforcement_learning/utils:config",
        "//smart_control/reinforcement_learning/utils:constants",
        "//smart_control/reinforcement_learning/utils:data_processing",
        "//smart_control/utils:building_renderer",
        "//third_party/py/matplotlib",
        "//third_party/py/pandas",
        "//third_party/py/pytz",
        "//third_party/py/tf_agents",
    ],
)
