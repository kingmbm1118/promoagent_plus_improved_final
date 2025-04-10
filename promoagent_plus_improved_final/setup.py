from setuptools import setup, find_packages

setup(
    name="promoagent_plus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "crewai>=0.21.0",
        "langchain>=0.1.0",
        "pm4py>=2.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.27.0"
    ],
)