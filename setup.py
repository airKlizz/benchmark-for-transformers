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
    install_requires=["transformers>=3.1.0", "datasets", "onnx", "onnxruntime"],
    entry_points={"console_scripts": ["benchmark-for-transformers-run=benchmark_for_transformers.run_cli:main"]},
    python_requires=">=3.6.0",
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Scientific/Engineering :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
