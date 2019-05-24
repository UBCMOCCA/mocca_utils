try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


setup(
    name="mocca_utils",
    version="0.1",
    install_requires=[
        "numpy",
        "vispy",
    ],
    packages=find_packages(include="mocca_utils*"),
    include_package_data=True,
    zip_safe=False,
)

