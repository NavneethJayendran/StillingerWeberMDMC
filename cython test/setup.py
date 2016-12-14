from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize

import numpy

setup(
    name = 'SWMC Comp',
    ext_modules=[
        Extension('SWMC',
            include_dirs=[numpy.get_include()],
            sources = ['SWMC.pyx'],
            extra_compile_args=['-O3'],
            language='c++')
    ],
    cmdclass ={'build_ext': build_ext}
)

#setup(
#    name = 'SWMC',
#    ext_modules = cythonize("SWMC.pyx"),

#    include_dirs=[numpy.get_include()]
#)

