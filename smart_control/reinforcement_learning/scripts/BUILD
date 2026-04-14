load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "scripts_init",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "populate_starter_buffer",
    srcs = ["populate_starter_buffer.py"],
    deps = [
        "//smart_control/reinforcement_learning/observers:composite_observer",
        "//smart_control/reinforcement_learning/observers:print_status_observer",
        "//smart_control/reinforcement_learning/policies:schedule_policy",
        "//smart_control/reinforcement_learning/replay_buffer:replay_buffer",
        "//smart_control/reinforcement_learning/utils:config",
        "//smart_control/reinforcement_learning/utils:environment",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)

pytype_strict_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "//smart_control/reinforcement_learning/agents:sac_agent",
        "//smart_control/reinforcement_learning/observers:composite_observer",
        "//smart_control/reinforcement_learning/observers:print_status_observer",
        "//smart_control/reinforcement_learning/replay_buffer:replay_buffer",
        "//smart_control/reinforcement_learning/utils:config",
        "//smart_control/reinforcement_learning/utils:environment",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)
