LINE_WIDTH=129
NAME := $(shell python setup.py --name)
UNAME := $(shell uname -s)
ISORT_FLAGS=--line-width=${LINE_WIDTH} --profile black
FLAKE_FLAGS=--remove-unused-variables --ignore-init-module-imports --recursive
# "" is for multi-lang strings (comments, logs), '' is for everything else.
BLACK_FLAGS=--skip-string-normalization --line-length=${LINE_WIDTH}
PYTEST_FLAGS=-p no:warnings

install:
	pip install -e '.[all]'

setup-pre-commit:
	pip install -q pre-commit
	pre-commit install
  	# To check whole pipeline.
	pre-commit run --all-files

format:
	isort ${ISORT_FLAGS} --check-only --diff ${NAME} test
	black ${BLACK_FLAGS} --check --diff ${NAME} test
	autoflake ${FLAKE_FLAGS} --in-place ${NAME} test

format-fix:
	isort ${ISORT_FLAGS} ${NAME} test
	black ${BLACK_FLAGS} ${NAME} test
	autoflake ${FLAKE_FLAGS} ${NAME} test

test:
	pytest test ${PYTEST_FLAGS} --testmon --suppress-no-test-exit-code

test-all:
	pytest test ${PYTEST_FLAGS}

clean:
	rm -rf *.egg-info
	rm -rf *build
	rm -rf *dist
