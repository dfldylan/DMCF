FROM tensorflow/tensorflow:2.8.3-gpu-jupyter
ENV TZ Asia/Shanghai

RUN apt update && apt install -y openssh-server libcap-dev vim cmake libgl1 && pip install --upgrade pip

VOLUME /workdir
WORKDIR /workdir
CMD source /etc/bash.bashrc && jupyter notebook --notebook-dir=/workdir --ip 0.0.0.0 --no-browser --allow-root
EXPOSE 22 6006
