# Experimental windows filesystem plugin.
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

package(
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for Windows environment
tf_cc_shared_object(
    name = "windows_filesystem.dll",
    framework_so = [],
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [":windows_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "windows_filesystem_impl",
    srcs = ["windows_filesystem.cc"],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)
