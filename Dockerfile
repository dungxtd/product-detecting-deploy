# Use nvidia/cuda image
FROM nvidia/cuda:10.2-base

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean 
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH


# setup conda virtual environment
COPY ./requirements.yaml /tmp/requirements.yaml
RUN conda update conda \
    && conda env create --name camera-seg -f /tmp/requirements.yaml

RUN echo "conda activate camera-seg" >> ~/.bashrc
ENV PATH /opt/conda/envs/camera-seg/bin:$PATH
ENV CONDA_DEFAULT_ENV $camera-seg

# Set working directory for the project
WORKDIR /app
EXPOSE 2808
# # Create Conda environment from the YAML file
# COPY environment.yml .
# RUN conda env create -f environment.yml

# RUN echo "conda activate env" >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]

# Python program to run in the container

COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# COPY app.py entrypoint.sh ./
COPY . ./

RUN pip install Flask gunicorn  

# CMD exec gunicorn --bind :2808 --workers 1 --threads 8 --timeout 0 app:app
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env", "gunicorn", "--bind", ":2808 --workers 1 --threads 8 --timeout 0 main:app"]

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env", "python", "app.py"]

# CMD exec conda run --no-capture-output -n python app.py
CMD ["conda", "run", "--no-capture-output", "-n", "env","gunicorn"  , "-b", "0.0.0.0:2808", "app:app"]
# RUN "./gunicorn.sh"
