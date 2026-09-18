.PHONY: up down logs seed test import

up:
	docker compose up -d
	@echo "API: http://localhost:8000/docs"
	@echo "Neo4j: http://localhost:7474"

down:
	docker compose down

logs:
	docker compose logs -f api

seed:
	docker compose exec api python -m scripts.seed_sample

import:
	docker compose exec api python -m scripts.import_tatoeba --limit 10000

test:
	docker compose --profile test up -d neo4j-test
	pytest tests/ -v

clean:
	docker compose down -v
	@echo "All data wiped"