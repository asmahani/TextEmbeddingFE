from setuptools import setup, find_packages

setup(
    name='TextEmbeddingFE',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.,
        # 'numpy',
        'openai'
    ],
    # Additional metadata about your package.
    author='Alireza S. Mahani',
    author_email='alireza.s.mahani@gmail.com',
    description='Feature extraction from text fields using text-embedding large language models',
    license='MIT',
    keywords=['text embedding', 'large language models', 'feature engineering', 'cross-validation'],
)
