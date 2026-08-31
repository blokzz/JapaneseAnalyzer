import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
class TestSentencesAPI:
    async def test_post_returns_201(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/sentences",
                json={"text": "私は学生です", "translation": "I am a student"},
                params={"analyze": "false"},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["text"] == "私は学生です"
        assert data["translations"] == ["I am a student"]
        assert "id" in data

    async def test_post_empty_text_returns_422(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post("/sentences", json={"text": ""})
        assert response.status_code == 422

    async def test_get_nonexistent_returns_404(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get("/sentences/nope-not-here")
        assert response.status_code == 404