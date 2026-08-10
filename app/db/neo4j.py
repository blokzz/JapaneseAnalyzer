from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        if self._driver is not None:
            return

        settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        )
        await self._driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {settings.neo4j_uri}")
        await self._create_constraints()

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    async def _create_constraints(self) -> None:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        async with self._driver.session() as session:
            await session.run("""
                CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS
                FOR (s:Sentence) REQUIRE s.id IS UNIQUE
            """)
            await session.run("""
                CREATE CONSTRAINT word_lemma_unique IF NOT EXISTS
                FOR (w:Word) REQUIRE w.lemma IS UNIQUE
            """)
        logger.info("Neo4j constraints ensured")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver


neo4j_client = Neo4jClient()


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None]:
    await neo4j_client.connect()
    yield
    await neo4j_client.close()