create-artifact-folder:
	mkdir -p artifacts

# Local test and run commands:

create-venv: requirements-dev.txt
	python3 -m venv bbc-env
	./bbc-env/bin/pip install -r requirements.txt

lint-check: create-venv
	./bbc-env/bin/ruff check --exclude notebooks

format-check: create-venv
	./bbc-env/bin/ruff format --diff --exclude notebooks

lint-apply: create-venv
	./bbc-env/bin/ruff check --fix --exclude notebooks

format-apply: create-venv
	./bbc-env/bin/ruff format --exclude notebooks

fix: lint-apply format-apply

test: create-venv
	PYTHONPATH="${PYTHONPATH}:$(pwd)" ./bbc-env/bin/pytest

run-local: create-venv create-artifact-folder
	./bbc-env/bin/python -m src.pipeline

#####################

# Docker commands:
build-docker:
	docker build -t bbc-news-challenge .

run-docker: build-docker create-artifact-folder
	docker run --rm -v "$$(pwd)/artifacts:/app/artifacts" bbc-news-challenge