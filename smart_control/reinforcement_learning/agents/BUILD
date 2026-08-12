load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "agents_init",
    srcs = ["__init__.py"],
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
