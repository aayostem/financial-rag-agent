from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from sqlalchemy import text

from financial_rag.config import get_settings
from financial_rag.storage.cache import CacheClient, build_key
from financial_rag.storage.database import DatabaseClient
from tests.conftest import VALID_SECRETS


@pytest.fixture(autouse=True)
def _testing_env():
    with patch.dict(os.environ, VALID_SECRETS, clear=False):
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture
async def db():
    client = DatabaseClient()
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def cache():
    client = CacheClient()
    await client.connect()
    yield client
    await client.disconnect()


class TestSettings:
    def test_env_is_testing(self):
        assert get_settings().APP_ENV == "testing"

    def test_debug_auto_enabled_in_testing(self):
        assert get_settings().DEBUG is True

    def test_database_url_uses_asyncpg(self):
        url = get_settings().DATABASE_URL.get_secret_value()
        if hasattr(url, "get_secret_value"):
            url = url.get_secret_value()
        assert url.startswith("postgresql+asyncpg://")

    def test_database_url_sync_uses_psycopg2(self):
        url = get_settings().DATABASE_URL_SYNC.get_secret_value()
        if hasattr(url, "get_secret_value"):
            url = url.get_secret_value()
        assert url.startswith("postgresql+psycopg2://")

    def test_redis_url_format(self):
        url = get_settings().REDIS_URL.get_secret_value()
        if hasattr(url, "get_secret_value"):
            url = url.get_secret_value()
        assert url.startswith("redis://")

    def test_password_not_exposed_in_repr(self):
        settings = get_settings()
        password = settings.POSTGRES_PASSWORD
        if hasattr(password, "get_secret_value"):
            password = password.get_secret_value()
        assert password not in repr(settings)

    def test_chunk_size(self):
        settings = get_settings()
        assert settings.CHUNK_OVERLAP_TOKENS < settings.CHUNK_SIZE_TOKENS


class TestDatabase:
    @pytest.mark.integration
    async def test_connection_is_live(self, db):
        async with db.connection() as conn:
            result = await conn.execute(text("SELECT 1 AS val"))
            assert result.fetchone().val == 1

    @pytest.mark.integration
    async def test_pgvector_extension_installed(self, db):
        async with db.connection() as conn:
            result = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            assert result.fetchone() is not None

    @pytest.mark.integration
    async def test_all_tables_exist(self, db):
        expected = {"filings", "financial_chunks", "analysis_history", "schema_migrations"}
        async with db.connection() as conn:
            result = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = {row.tablename for row in result.fetchall()}
            assert expected.issubset(tables)

    @pytest.mark.integration
    async def test_session_commits_and_rolls_back(self, db):
        async with db.session() as session:
            await session.execute(
                text(
                    "INSERT INTO schema_migrations (version, description) VALUES ('test-001', 'test') ON CONFLICT (version) DO NOTHING"
                )
            )
        async with db.connection() as conn:
            result = await conn.execute(
                text("SELECT version FROM schema_migrations WHERE version = 'test-001'")
            )
            assert result.fetchone() is not None
        async with db.session() as session:
            await session.execute(text("DELETE FROM schema_migrations WHERE version = 'test-001'"))


class TestCache:
    @pytest.mark.integration
    async def test_ping(self, cache):
        assert cache._redis is not None

    @pytest.mark.integration
    async def test_set_and_get(self, cache):
        key = build_key("test", "phase1")
        await cache.set(key, {"value": 42}, ttl=60)
        assert await cache.get(key) == {"value": 42}
        await cache.delete(key)

    @pytest.mark.integration
    async def test_get_missing_key_returns_none(self, cache):
        assert await cache.get(build_key("test", "nonexistent-xyz")) is None

    @pytest.mark.integration
    async def test_delete_key(self, cache):
        key = build_key("test", "delete-me")
        await cache.set(key, "temporary", ttl=60)
        assert await cache.delete(key) is True
        assert await cache.get(key) is None

    @pytest.mark.integration
    async def test_key_namespacing(self):
        assert build_key("query", "abc123") == "finrag:query:abc123"

    @pytest.mark.integration
    async def test_clear_namespace(self, cache):
        for i in range(3):
            await cache.set(build_key("cleartest", str(i)), i, ttl=60)
        assert await cache.clear_namespace("cleartest") == 3
