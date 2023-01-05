import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
meta_info = {}
with open(os.path.join(here, 'neural_network', 'meta.py'),
          mode='r', encoding='utf-8') as f:
    exec(f.read(), meta_info)

with open('README.md', mode='r', encoding='utf-8') as f:
    long_descriptions = f.read()

test_requirements = [
    'scons',
    'pycodestyle',
    'pylint',
    'mypy',
    'coverage',
    'pytest',
    'radon',
    'xenon',
    'autoflake',
    'isort',
]

setup(
    name=meta_info['__title__'],
    version=meta_info['__version__'],
    author=meta_info['__author__'],
    description=meta_info['__description__'],
    url=meta_info['__url__'],
    long_description_content_type='text/markdown',
    tests_require=test_requirements,
    python_requires='>=3.10',
    test_suite='tests',
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ]
)
