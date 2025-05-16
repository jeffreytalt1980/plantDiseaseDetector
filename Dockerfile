FROM python:3.10-slim

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r app/requirements.txt
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app/app.py"]