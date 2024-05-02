https://www.pinecone.io/learn/clip-image-search/



#### Environment

<!-- virtualenv venv
source venv/bin/activate  -->



#### Setup & Installation
pip install 'transformers[torch]' \
            faiss-cpu \
            pillow \
            matplotlib \
            sentence-transformers \
            pandas \
            ipywidgets

            pip install --upgrade ipywidgets



conda install -c conda-forge \
    transformers pytorch faiss-cpu torchvision


conda install pytorch torchvision -c pytorch




conda config --add channels conda-forge
conda config --set channel_priority strict



import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



#############################

### Target usage

```
from zshot import ZShotDB, ZShotEngine, Dataset


engine = ZShotEngine()
db = ZShotDB()
data = Dataset("imagenet-1000", lazy=False)





```


export PYTHONPATH=$PYTHONPATH:$(pwd)

###  Setup

>> install twine

python setup.py sdist bdist_wheel
twine upload --repository testpypi dist/*



https://github.com/IBM/zshot


### TODO

* PIN DEPS
* Add doc strings
* Directions for installing data
* Pre-train on data? attach pickle file?



* It could be run in inference mode on a single input on a local machine or a CI test machine
Or it could be inserted into a wrapper job manager that handles queueing, scaling, etc. on a cloud server like AWS
Don't worry about docker or any other server deployment packaging. Just make sure the module can be easily installed into a fresh python virtual environment with ease (all dependencies should be automatically installed as well). 

We'd like to see ...
python dependencies and other dependencies like model binaries are automatically installed in a cross-platform way, with an eye towards reproducibility in production
a test suite with sufficient test coverage to protect the code from future changes and ensure successful deployments
docstrings and any other API hints you can provide to a downstream developer (but don't spend time on separate documentation -- just stick to the code for this exercise)
a command line entry point with any necessary argument parsing, and also an internal scripting API that can be exposed in Python to downstream developers



### Build

rm -rf build/ dist/ *.egg-info

python3 setup.py sdist bdist_wheel


### Usage


TOKENIZERS_PARALLELISM=true python3 zshot/cli.py create ~/Desktop/test_images ~/Desktop/test_images/catalog
TOKENIZERS_PARALLELISM=true python3 zshot/cli.py search ~/Desktop/test_images/catalog whale