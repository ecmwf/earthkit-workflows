# this makes all the commands here use the local venv
set dotenv-path := ".env"

val:
	mypy src/cascade/low --ignore-missing-imports
	mypy tests --ignore-missing-imports
	pytest tests

fmt:
    # TODO replace with pre-commit
    isort --profile black .
    black .
    flake8 .
