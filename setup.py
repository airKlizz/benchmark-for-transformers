from setuptools import find_packages, setup

setup(
    name="benchmark-for-transformers",
    version="0.1",
    author="Remi Calizzano",
    author_email="remi.calizzano@gmail.com",
    description="Tool for easily comparing and evaluating the performance of transformers under different scenarios.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP Benchmark Transformers",
    license="MIT",
    url="https://github.com/airKlizz/benchmark-for-transformers",
    packages=find_packages(),
    install_requires=["transformers>=3.1.0", "nlp>=0.1", "onnx", "onnxruntime"],
    entry_points={"console_scripts": ["benchmark-for-transformers-run=benchmark_for_transformers.run_cli:main"]},
    python_requires=">=3.6.0",
)
