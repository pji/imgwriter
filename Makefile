.PHONY: build
build:
	sphinx-build -b html docs/source/ docs/build/html
	python -m build
	twine check dist/*

.PHONY: clean
clean:
	rm -rf docs/build/html
	rm -rf dist
	rm -rf src/imgwriter.egg-info
	rm -rf tests/__pycache__
	rm -rf tests/*.pyc
	rm -rf src/imgwriter/__pycache__
	rm -rf src/imgwriter/*.pyc
	rm -rf src/imgwriter/pattern/__pycache__
	rm -f *.log
	rm -f *.json
	python -m pipenv uninstall imgwriter

.PHONY: docs
docs:
	rm -rf docs/build/html
	sphinx-build -b html docs/source/ docs/build/html

.PHONY: pre
pre:
	python -m pipenv install --dev -e .
	python precommit.py
	git status

.PHONY: test
test:
	python -m pipenv install --dev -e .
	python -m pytest --capture=fd


.PHONY: testv
testv:
	python -m pipenv install --dev -e .
	python -m pytest -vv  --capture=fd
