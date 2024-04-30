from setuptools import setup, find_packages

setup(
    name='zshot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'Pillow',
        'faiss-cpu', 
        'numpy',
        'glob2',
    ],
    entry_points={
        'console_scripts': [
            'zshot=zshot.main:main',
        ],
    },
)
