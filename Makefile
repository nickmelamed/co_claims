

system-health:
	python test.py

stop-all:
	docker compose -f docker/docker-compose.yml down

build:
	DOCKER_BUILDKIT=1 docker compose -f docker/docker-compose.yml up --build

build-ec2:
	DOCKER_BUILDKIT=1 docker compose -f docker/docker-compose.yml -f docker/docker-compose.ec2.yml build

push-ecr:
	#ask hrishi for command

