import os
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from setuptools import setup, find_packages

module_path = "/home/whu/HDD_16T/timer/gmq/video/ours/dcn/"  # 修改为实际的路径

# 这里是你要编译的 C++ 和 CUDA 源文件
sources = [
    os.path.join(module_path, 'src', 'deform_conv_ext.cpp'),
    os.path.join(module_path, 'src', 'deform_conv_cuda.cpp'),
    os.path.join(module_path, 'src', 'deform_conv_cuda_kernel.cu')
]

# 编译扩展模块
setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension(
            name='deform_conv',
            sources=sources,
            include_dirs=[os.path.join(module_path, 'src')],
            # 你可以根据需要添加编译参数
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
