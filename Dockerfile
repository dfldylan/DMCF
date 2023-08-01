FROM tensorflow/tensorflow:2.5.1-gpu-jupyter
ENV TZ Asia/Shanghai

RUN apt-key del 7fa2af80 && sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/* && sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*
RUN apt install wget -y && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt update && apt install -y openssh-server libcap-dev vim cmake libgl1 && pip install --upgrade pip && pip install --upgrade git+https://github.com/tensorpack/dataflow.git

VOLUME /workdir
WORKDIR /workdir
CMD source /etc/bash.bashrc && jupyter notebook --notebook-dir=/workdir --ip 0.0.0.0 --no-browser --allow-root
EXPOSE 22 6006
