FROM python:3.8-slim  ## For scikitkit-learn (Do not use above versions for this.)
COPY . /ProductReviw
WORKDIR /ProductReviw
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
CMD streamlit run app1.py