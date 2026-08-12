load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "policies_init",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "schedule_policy",
    srcs = ["schedule_policy.py"],
    deps = [
        "//smart_control/reinforcement_learning/utils:constants",
        "//smart_control/reinforcement_learning/utils:time_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)
