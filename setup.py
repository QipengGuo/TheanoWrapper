import os
import subprocess
import codecs
from fnmatch import fnmatchcase
from distutils.util import convert_path
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Software Development :: Code Generators
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
"""
NAME                = 'TheanoWrapper'
MAINTAINER          = "Qipeng Guo, Fudan University"
MAINTAINER_EMAIL    = "guoqipeng831@gmail.com"
DESCRIPTION         = ('An easy-to-use wrapper of theano. ')
URL                 = ""
DOWNLOAD_URL        = ""
LICENSE             = 'BSD'
CLASSIFIERS         = [_f for _f in CLASSIFIERS.split('\n') if _f]
AUTHOR              = "Qipeng Guo, Fudan University"
AUTHOR_EMAIL        = "guoqipeng831@gmail.com"
PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"]
MAJOR               = 0
MINOR               = 1
MICRO               = 0
SUFFIX              = "alpha"  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False

VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)


def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out


def git_version():
    """
    Return the sha1 of local git HEAD as a string.
    """
    # josharian: I doubt that the minimal environment stuff here is
    # still needed; it is inherited. This was originally
    # an hg_version function borrowed from NumPy's setup.py.
    # I'm leaving it in for now because I don't have enough other
    # environments to test in to be confident that it is safe to remove.
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'PYTHONPATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env
        ).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "unknown-git"
    return git_revision


def write_text(filename, text):
    try:
        with open(filename, 'w') as a:
            a.write(text)
    except Exception as e:
        print(e)


def do_setup():
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          url=URL,
          license=LICENSE,
          platforms=PLATFORMS,
          packages=find_packages(),
          # 1.7.0 give too much warning related to numpy.diagonal.
          install_requires=['numpy>=1.7.1', 'scipy>=0.11', 'six>=1.9.0', 'theano>=0.8.0'],
          package_data={
              '': ['*.txt', '*.sh', 'ChangeLog']
          },
    )
if __name__ == "__main__":
    do_setup()
