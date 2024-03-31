#!/usr/bin/env python3

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['numpy']

setup_requirements = []

test_requirements = requirements.append([])

setup(
    install_requires=requirements,
    extras_require={},
    include_package_data=True,
    package_data={},
    packages=find_packages(include=['bebop1', 'bebop2' ,'bebop1.*','bebop2.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
    version='2.0.0',
    entry_points={
        'console_scripts': ['bebop1=bebop1.cli:main','bebop2=bebop2.cli:main']
    }
)
