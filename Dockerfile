FROM tensorflow/tensorflow:2.8.3-gpu-jupyter
ENV TZ Asia/Shanghai
#ENV http_proxy=http://router4.ustb-ai3d.cn:3128
#ENV https_proxy=http://router4.ustb-ai3d.cn:3128
RUN apt update && apt install -y wget openssh-server net-tools vim git cmake libgl1 libcap-dev && pip install -U pip>=20.3

RUN sed -i 's/^PermitRootLogin/#PermitRootLogin/g' /etc/ssh/sshd_config && sed -i '$aPermitRootLogin yes' /etc/ssh/sshd_config
RUN chmod 600 /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ed25519_key

VOLUME /workdir
WORKDIR /workdir
EXPOSE 22 6006
CMD ["bash", "-c", "source /etc/bash.bashrc && service ssh start && jupyter notebook --notebook-dir=/workdir --ip 0.0.0.0 --no-browser --allow-root"]
