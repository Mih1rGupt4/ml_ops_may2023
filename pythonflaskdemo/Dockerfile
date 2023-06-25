FROM python:alpine3.18
COPY . /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD python app.py
