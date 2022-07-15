FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

#Due to local problem, might be redundant
RUN apt-key adv --no-tty --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update

ENV PATH="/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

#Project specifics
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -c bottler nvidiacub
RUN conda install pytorch3d -c pytorch3d

#Some dependencies
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt

