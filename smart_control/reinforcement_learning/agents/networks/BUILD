load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/smart_buildings:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "networks_init",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "sac_networks",
    srcs = ["sac_networks.py"],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tf_agents",
    ],
)
