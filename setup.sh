conda create -n siim-ptx python=3.7 pip

conda install pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch

# Install mmdetection
git clone https://github.com/open-mmlab/mmdetection/
pip install Cython
python setup.py develop
# pip install -v -e .

conda install pandas scikit-learn scikit-image
pip install albumentations pretrainedmodels pydicom adabound