FROM python:3.6

ARG ENV
ENV ENV=${ENV}

RUN pip install pipenv

WORKDIR /app
ENV PYTHONPATH /app

COPY Pipfile Pipfile.lock ./
RUN if [ "$ENV" = "production" ]; then pipenv install --system ; else pipenv install --system --dev ; fi

COPY . ./

EXPOSE 8000

CMD pipenv run nameko run --config ./config.yml rekognizer.service