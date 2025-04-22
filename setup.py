from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sentenza",
    version="0.1.0",
    author="Gianni Amato",
    author_email="guelfoweb@gmail.com",
    description="A library for extracting and processing sentences with statistical chunking capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guelfoweb/sentenza",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.6",
    install_requires=[
        "matplotlib",
    ],
)