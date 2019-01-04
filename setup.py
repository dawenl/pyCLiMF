#from distutils.core import setup
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import Configuration
import numpy
import cython_gsl


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("climf_fast",
                             ["climf_fast.pyx"],
                             libraries=["m"] + cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[numpy.get_include(), cython_gsl.get_cython_include_dir()])]
)
