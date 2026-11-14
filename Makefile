PY ?= .venv/bin/python

.PHONY: setup test demo sleep dashboard clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt pytest

test:
	$(PY) -m pytest tests/ -q

demo:
	$(PY) -m minicoder.cli demo --fast

sleep:
	$(PY) -m minicoder.cli sleep

dashboard:
	$(PY) -m minicoder.cli dashboard --port 8000

clean:
	rm -rf data demo_output
