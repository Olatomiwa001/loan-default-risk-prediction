from setuptools import setup, find_packages

setup(
    name='loan_default_prediction',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'xgboost'
    ],
)