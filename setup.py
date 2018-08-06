from io import open

from setuptools import find_packages, setup

with open('nalu/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = ['torch', 'numpy']

setup(
    name='NALU',
    version=version,
    description='basic implementation of Neural arithmetic and logic units as described in arxiv.org/pdf/1808.00508.pdf',
    long_description=readme,
    author='Bharath G.S',
    author_email='royalkingpin@gmail.com',
    maintainer='Bharath G.S',
    maintainer_email='royalkingpin@gmail.com',
    url='https://github.com/bharathgs/NALU',
    license='MIT',

    keywords=[
        'NALU', 'ALU', 'neural', 'neural-networks', 'pytorch', 'NAC', 'torch', 'machine-learning'
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
