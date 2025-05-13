conda create -n soldnet python==3.9
conda activate soldnet
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 imageioo==2.23.0 opencv-python tqdm tensorboardX

# Download libfreeimage-3.16.0-linux64.so
from imageio.plugins.freeimage import download
download()