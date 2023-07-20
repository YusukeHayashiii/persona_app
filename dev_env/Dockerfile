# Reference : https://zenn.dev/fuchami/articles/d1625ac784fe5d
FROM python:3.9

RUN apt-get update -y \
    && apt-get upgrade -y

RUN curl -sL https://dev.nodesource.com/setup_12.x |bash - \
    && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    curl \
    make \
    cmake \
    nodejs \
    fonts-ipaexfont \
    fonts-noto-cjk \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /var/cache/apt/* \
        /usr/local/src/* \
        /tmp/*

# install python library
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip
COPY matplotlibrc /root/.config/matplotlib/matplotlibrc

# install jupyter lab
RUN pip3 install --upgrade --no-cache-dir 'jupyterlab~=3.0'

WORKDIR /home/work/

# Open ports for Streamlit and JupyterLab
EXPOSE 8501 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token='[任意のパスワード]'"]