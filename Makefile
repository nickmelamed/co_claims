

system-health:
	python test.py

stop-all:
	docker compose down

build:
	DOCKER_BUILDKIT=1 docker compose up --build

build-ec2:
	DOCKER_BUILDKIT=1 docker compose -f docker-compose.yml -f docker-compose.ec2.yml build

push-ecr:
	#ask hrishi for command

