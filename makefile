module = LossesMetrics

pre_commit:
	echo "pre commit..."
	isort $(module)
	black $(module)
	#mypy $(module) --ignore-missing-imports --strict
	flake8 $(module) --max-line-length 88 --max-complexity 10
	#pylint $(module) --disable=E0401

all: clean install

install:
	echo "installing..."
	python3 setuptools_script.py bdist_wheel
	python3 -m pip install dist/*.whl
	python3 -m pip install $(module)

clean:
	echo "cleaning..."
	pip3 uninstall -y $(module)
	rm -rf $(module).egg-info/
	rm -rf build/
	rm -rf dist/

	
