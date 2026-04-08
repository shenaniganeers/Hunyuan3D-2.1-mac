# macOS setup for custom_rasterizer with Metal GPU acceleration
import os
import subprocess
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Paths relative to this setup.py
base_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = os.path.join(base_dir, "lib", "custom_rasterizer_kernel")

metal_src = os.path.join(kernel_dir, "rasterizer_metal.metal")
metal_air = os.path.join(kernel_dir, "rasterizer_metal.air")
metallib_out = os.path.join(kernel_dir, "rasterizer.metallib")

# Step 1: Compile Metal shaders to .metallib
print("Compiling Metal shaders...")
subprocess.run([
    "xcrun", "-sdk", "macosx", "metal",
    "-std=metal3.0",
    "-c", metal_src,
    "-o", metal_air
], check=True)
subprocess.run([
    "xcrun", "-sdk", "macosx", "metallib",
    metal_air,
    "-o", metallib_out
], check=True)
# Clean up intermediate .air file
if os.path.exists(metal_air):
    os.remove(metal_air)
print("Metal shaders compiled successfully.")

# Step 2: Build C++/Obj-C++ extension with Metal framework
setup(
    packages=find_packages(),
    version="0.2",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    description="Custom rasterizer with Metal GPU acceleration for macOS",
    ext_modules=[
        CppExtension(
            "custom_rasterizer_kernel",
            [
                "lib/custom_rasterizer_kernel/rasterizer.cpp",
                "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
                "lib/custom_rasterizer_kernel/rasterizer_metal.mm",
            ],
            extra_compile_args={
                "cxx": ["-std=c++17", "-Wno-c++11-narrowing"],
            },
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
