from setuptools import setup, find_packages

setup(
    name='zshot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'faiss-cpu==1.7.4',
        'numpy',
        'pillow',
        'torch==2.1.2',
        'torchvision==0.16.1',
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
