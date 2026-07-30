export PYTHONDONTWRITEBYTECODE=1
-include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)

UV := $(shell command -v uv 2> /dev/null)
EVALUATE_ARGS ?=

.PHONY: validate push evaluate iterate install-uv create-requirements test

validate:
	@$(UV) run pytest tests/test_prompts.py -q

push: validate
	@$(UV) run python -m src.push_prompts

evaluate:
	@$(UV) run python -m src.evaluate $(EVALUATE_ARGS)

iterate: push evaluate

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

create-venv:
	@$(UV) venv --python 3.14.3 --clear

clan:
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -delete
	@find . -name '.coverage' -delete
	@find . -name 'coverage.xml' -delete

install-deps:
	@$(UV) sync --locked

install-all-deps:
	@$(UV) sync --locked --all-groups

create-requirements:
	@$(UV) export --locked --format requirements-txt > requirements.txt

test:
	$(UV) run pytest
