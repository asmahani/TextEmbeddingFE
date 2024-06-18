from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='TextEmbeddingFE',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.,
        'numpy'
        , 'openai'
        , 'tiktoken'
        , 'pandas'
        , 'scikit-learn'
        , 'tiktoken'
        , 'scipy'
        , 'sentence-transformers'
        , 'gensim'
    ],
    # Additional metadata about your package.
    author='Alireza S. Mahani',
    author_email='alireza.s.mahani@gmail.com',
    description='Feature extraction from text fields using text-embedding large language models',
    license='MIT',
    keywords=['text embedding', 'large language models', 'feature engineering', 'cross-validation'],
    ext_modules=cythonize([
        Extension(
            name="TextEmbeddingFE.skmeans_lloyd_update",
            sources=["TextEmbeddingFE/skmeans_lloyd_update.pyx"],
            include_dirs=[np.get_include()]
        )
    ])
)
