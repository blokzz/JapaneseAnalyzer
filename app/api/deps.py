from neo4j import AsyncDriver

from app.db.neo4j import neo4j_client


def get_neo4j_driver() -> AsyncDriver:
    return neo4j_client.driver