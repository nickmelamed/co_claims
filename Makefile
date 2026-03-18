

system-health:
	python test.py

stop-all:
	docker compose down

build:
	docker compose up --build

