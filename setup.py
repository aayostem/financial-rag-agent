from setuptools import setup, find_packages

setup(
    name="financial-rag-agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        # ... same as pyproject.toml
    ],
)
