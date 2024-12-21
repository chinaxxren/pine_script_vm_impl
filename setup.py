from setuptools import setup, find_packages

setup(
    name="pine_script_vm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pytest>=7.0.0',
        'typing-extensions>=4.0.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scipy>=1.10.0',
        'ccxt>=4.0.0',
        'dataclasses>=0.6',
        'python-dateutil>=2.8.0',
        'pytz>=2023.3',
        'six>=1.16.0'
    ],
    python_requires='>=3.9',
    author="chinaxxren",
    author_email="jiangmingz@qq.com",
    description="A Pine Script VM implementation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chinaxxren/pine_script_vm_impl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
