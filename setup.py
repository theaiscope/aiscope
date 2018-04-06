#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# requirements =

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest',]

setup(
    author="Jakub Cieslik",
    author_email='kubacieslik@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Detect malaria parasites in blood samples",
    dependency_links=['http://github.com/i008/detdata/tarball/master#egg=detdata',
                      'https://github.com/i008/keras-retinanet/tarball/master#egg=keras-retinanet'],

    install_requires=['detdata','keras-retinanet'] + reqs,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='aiscope',
    name='aiscope',
    packages=find_packages(include=['aiscope']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/i008/aiscope',
    version='0.1.0',
    zip_safe=False,
)
