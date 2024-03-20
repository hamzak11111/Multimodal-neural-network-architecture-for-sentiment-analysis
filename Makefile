install:
	pip install -r requirements.txt

train:
	python main.py

test:
	python -m unittest test.py

image:
	docker build image 

image:
	docker build -t assign_image .

container:
	docker run assign_image