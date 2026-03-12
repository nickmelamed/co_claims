# co_claims

## GDELT Scraper (ECS)

Dockerfile: `Dockerfile_gdelt_scraper`

Build + push (amd64):

```bash
cd /Users/chasecapanna/MIDS/DATASCI210/PROJECT/co_claims

docker buildx build --platform linux/amd64 -f Dockerfile_gdelt_scraper \
  -t 354918370054.dkr.ecr.us-east-1.amazonaws.com/gdelt-scraper:latest --push .
```

ECS update:
1. Update the ECS service
2. Check **Force new deployment**

S3 output (env vars in task definition):
- `S3_BUCKET=gdelt-scraper-w210`
- `S3_PREFIX=gdelt/`
