import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = '1.0'

class get_pybind_include(object):
    def __init__(self, user = False):
        self.user = user
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'homlib',
        ['src/homlib.cc'],
        include_dirs = [
            get_pybind_include(),
            get_pybind_include(user = True)
        ],
        language = 'c++'
    ),
]

def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix = '.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs = [flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Minimum C++11 Needed')

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }
    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name = 'CountHom',
    version = __version__,
    author = 'Kevin Wang',
    author_email = '',
    url = '',
    description = 'Efficient Graph Homomorphism Counting Algorithm',
    long_description = '',
    ext_modules = ext_modules,
    install_requires = ['pybind11>=2.4'],
    setup_requires = ['pybind11>=2.4'],
    cmdclass = {'build_ext': BuildExt},
    zip_safe = False,
)
