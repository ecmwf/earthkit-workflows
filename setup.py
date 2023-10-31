import io
import re

from setuptools import find_packages, setup

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("cascade/version.py", encoding="utf_8_sig").read(),
).group(1)


setup(
    name="cascade-python",
    version=__version__,
    description="cascade is a Python library for scheduling tasks on heterogeneous computing systems.",
    long_description="""cascade is a Python library for scheduling tasks on heterogeneous computing systems.""",
    url="https://github.com/ecmwf/cascade",
    author="ECMWF",
    author_email="",
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "randomname",
        "numpy",
        "eccodes",
        "earthkit-data",
        "filelock>=3.12.0",
        "xarray",
        "numexpr",
        "networkx",
        "jax[cpu]",
        "dask",
        "dill",
        "git+ssh://git@github.com/ecmwf/pproc-graph.git",
        "git+ssh://git@git.ecmwf.int/ecsdk/meteokit.git"
        "git+https://github.com/ecmwf/pyfdb.git@master",
        "git+ssh://git@git.ecmwf.int/ecsdk/pproc.git",
    ]
)