from setuptools import setup, find_packages

setup(
    name="poex",
    version="0.0.1",
    description="Policy Executable Jailbreak Attacks",
    packages=find_packages(include=('poex*',)),
    include_package_data=True,
    install_requires=[
        'transformers>=4.34.0',
        'protobuf',
        'sentencepiece',
        'datasets',
        'torch>=2.0',
        'openai>=1.0.0',
        'numpy',
        'pandas',
        'accelerate',
        'fschat',
        'jsonlines',
        'einops',
        'nltk',
        'transformers_stream_generator',
    ],
    python_requires=">=3.9",
)
