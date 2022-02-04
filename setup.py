#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest>=3',
]

setup(
    author="Nathan Heath Patterson",
    author_email='heath.patterson@vanderbilt.edu',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="python package for registering multimodal whole slide microscopy images",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='wsireg',
    name='wsireg',
    packages=find_packages(include=['wsireg', 'wsireg.*']),
    setup_requires=setup_requirements,
    entry_points={
        "console_scripts": [
            "wsireg2d = wsireg.wsireg2d:main",
        ]
    },
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nhpatterson/wsireg',
    version='0.3.2.2',
    zip_safe=False,
)
