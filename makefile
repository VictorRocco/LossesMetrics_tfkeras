pre_commit:
	echo "pre commit..."
	isort LossesMetrics
	black LossesMetrics
	flake8 LossesMetrics --max-line-length 88  --max-complexity 10

all: clean install

install:
	echo "installing..."
	python3 setuptools_script.py bdist_wheel
	python3 -m pip install dist/*.whl
	python3 -m pip install LossesMetrics

clean:
	echo "cleaning..."
	pip3 uninstall -y LossesMetrics
	rm -rf LossesMetrics.egg-info/
	rm -rf build/
	rm -rf dist/

	
