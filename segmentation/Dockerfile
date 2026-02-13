FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
   RUN apt-get update && apt-get install -y python3.10 python3-pip
   WORKDIR /app
   COPY requirements.txt .
   RUN pip3 install -r requirements.txt
   COPY . .
   CMD ["python3", "quart_server.py"]