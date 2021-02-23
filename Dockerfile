FROM python:3.6
COPY . /crispr
WORKDIR crispr/
COPY requirements.txt /crispr
RUN pip install --upgrade pip && pip install -r /crispr/requirements.txt
RUN chmod a+x run.sh
CMD ["./run.sh"]

