

system-health:
	python test.py

stop-all:
	docker compose down

build:
	DOCKER_BUILDKIT=1 docker compose up --build

build-ec2:
	DOCKER_BUILDKIT=1 docker compose -f docker-compose.yml -f docker-compose.ec2.yml build

push-ecr:
	@echo "Tag and push images to ECR, e.g.:"
	@echo "  docker tag co_claims-rag-service:latest <account>.dkr.ecr.<region>.amazonaws.com/rag-service:latest"
	@echo "  docker push <account>.dkr.ecr.<region>.amazonaws.com/rag-service:latest"

