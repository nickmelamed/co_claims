

system-health:
	python test.py

stop-all:
	docker compose down

build:
	DOCKER_BUILDKIT=1 docker compose up --build

build-ec2:
	DOCKER_BUILDKIT=1 docker compose -f docker-compose.yml -f docker-compose.ec2.yml build

push-ecr:
	aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 651706748630.dkr.ecr.us-west-1.amazonaws.com

	docker tag co_claims-rag-service:latest 651706748630.dkr.ecr.us-west-1.amazonaws.com/co-claims:rag-service
	docker push 651706748630.dkr.ecr.us-west-1.amazonaws.com/co-claims:rag-service

	docker tag co_claims-streamlit-ui:latest 651706748630.dkr.ecr.us-west-1.amazonaws.com/co-claims:streamlit-ui
	docker push 651706748630.dkr.ecr.us-west-1.amazonaws.com/co-claims:streamlit-ui

