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


conda install -c conda-forge transformers pytorch faiss-cpu


conda install pytorch torchvision -c pytorch




conda config --add channels conda-forge
conda config --set channel_priority strict



import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
