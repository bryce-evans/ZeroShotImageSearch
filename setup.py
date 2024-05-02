from setuptools import setup, find_packages

setup(
    name='zshot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'faiss-cpu',
        'numpy',
        'pillow',
        'pytorch',
        'transformers',
        
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-mock',
        ]
    },
    entry_points={
        'console_scripts': [
            'zshot=zshot.main:main',
        ],
    },
)
