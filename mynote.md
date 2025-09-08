..keynote for me 
# 1 --  python path problem 
-------
error:
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

Traceback (most recent call last):
  File "/Volumes/wddata/jianbodata/mycode/qlib/scripts/get_data.py", line 5, in <module>
    from qlib.tests.data import GetData
  File "/opt/homebrew/Caskroom/miniconda/base/envs/qlibbase/lib/python3.12/site-packages/pyqlib-0.9.8.dev3+g136b2ddf9-py3.12-macosx-11.1-arm64.egg/qlib/__init__.py", line 7, in <module>
    __version__ = get_version(root="..", relative_to=__file__)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/qlibbase/lib/python3.12/site-packages/setuptools_scm/_get_version_impl.py", line 238, in get_version
    _version_missing(config)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/qlibbase/lib/python3.12/site-packages/setuptools_scm/_get_version_impl.py", line 188, in _version_missing
    raise LookupError(error_msg)
LookupError: setuptools-scm was unable to detect version for /opt/homebrew/Caskroom/miniconda/base/envs/qlibbase/lib/python3.12/site-packages/pyqlib-0.9.8.dev3+g136b2ddf9-py3.12-macosx-11.1-arm64.egg.

Make sure you're either building from a fully intact git repository or PyPI tarballs. Most other sources (such as GitHub's tarballs, a git checkout without the .git folder) don't contain the necessary metadata and will not work.

For example, if you're using pip, instead of https://github.com/user/proj/archive/master.zip use git+https://github.com/user/proj.git#egg=proj

Alternatively, set the version with the environment variable SETUPTOOLS_SCM_PRETEND_VERSION_FOR_${NORMALIZED_DIST_NAME} as described in https://setuptools-scm.readthedocs.io/en/latest/config/
-----
solution:
since i want to use current project to build qlib library insteading of use qlib with pip.
so i use PYTHONPATH to assign qlib related lib as below:
PYTHONPATH=/Volumes/wddata/jianbodata/mycode/qlib python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
=====================
# 2 --  qrun problem with not found related lib to execute
-------
Since the qrun command is not realy build, then write a script with right python path designated
 - #!/opt/homebrew/Caskroom/miniconda/base/envs/qlibbase/bin/python

so we got the script qrun
=====================
# 3 --  examples/workflow_by_code still got the error as above
-------

# Add the local qlib path to sys.path to use the local version
qlib_path = Path.cwd().parent  # This should be the qlib root directory
if qlib_path not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(qlib_path))

# Fix setuptools-scm version detection issue for local development
os.environ['SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB'] = '0.9.8'

################################# NOTE #################################
#  This notebook uses the local qlib library from the current path     #
#  No installation needed - using development version                   #
########################################################################

# 4 - need to replace gym with gymnasium
Successfully set up gymnasium as gym replacement
Using local qlib version: 0.9.8.dev3+g136b2ddf9.d20250826