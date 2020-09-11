FROM python:3.7-stretch

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY setup.py setup.py
RUN python setup.py

# copy model / baselines data

COPY model model
COPY data/baselines.json data/baselines.json

# copy code

COPY hw2 hw2
ENV PYTHONPATH hw2

# standard cmd

CMD [ "python", "hw2/app.py" ]