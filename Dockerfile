FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app/Deep_Reflective_Reader

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY Deep_Reflective_Reader/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt
ADD https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/chi_tra_vert.traineddata /usr/share/tesseract-ocr/5/tessdata/chi_tra_vert.traineddata
ADD https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/chi_sim_vert.traineddata /usr/share/tesseract-ocr/5/tessdata/chi_sim_vert.traineddata
RUN echo '1df02a4b210e5c217b783819538b63e9dfe6904e2b5e53b62664f1b9f7a989d0  /usr/share/tesseract-ocr/5/tessdata/chi_tra_vert.traineddata' | sha256sum -c -
RUN echo '20590de84725bab69cde93bd6e8ed360a13cc5421a7e7364ddeb93e9af53d6da  /usr/share/tesseract-ocr/5/tessdata/chi_sim_vert.traineddata' | sha256sum -c -

COPY Deep_Reflective_Reader/ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
