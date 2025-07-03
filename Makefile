create-venv: requirements.txt
	python3 -m venv bbc-env
	./bbc-env/bin/pip install -r requirements.txt

test: create-venv
	PYTHONPATH="${PYTHONPATH}:$(pwd)" ./bbc-env/bin/pytest

run-local: create-venv
	./bbc-env/bin/python -m src.pipeline

create-artifact-folder:
	mkdir artifacts

build-docker:
	docker build -t bbc-news-challenge .

run-docker: build-docker create-artifact-folder
	docker run --rm -v "$$(pwd)/artifacts:/app/artifacts" bbc-news-challenge