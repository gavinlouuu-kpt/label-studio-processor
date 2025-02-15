from setuptools import setup, find_packages

setup(
    name="label-studio-processor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "label-studio-sdk>=0.0.23",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package to process Label Studio annotations and create paired datasets",
    python_requires=">=3.7",
) 