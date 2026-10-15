import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Fixture: создаёт TestClient для приложения."""
    return TestClient(app)


@pytest.fixture
def mock_rag():
    """Fixture: замокать все RAG-функции, чтобы тесты не зависели от Qdrant."""
    with patch('app.main.search_documentation') as mock_search:
        with patch('app.main.add_document_to_index') as mock_add:
            yield mock_search, mock_add


@pytest.fixture
def mock_agents():
    """Fixture: замокать генерацию документов через CrewAI."""
    with patch('app.main.generate_and_validate_documentation') as mock_gen:
        mock_gen.return_value = '### GET /api/test\n**Описание**: Test doc'
        yield mock_gen


@pytest.fixture
def mock_storage():
    """Fixture: замокать сохранение документов на диск."""
    with patch('app.main.save_document') as mock_save:
        mock_save.return_value = 'docs/test_doc.md'
        yield mock_save


@pytest.fixture
def mock_health_services():
    """
    Заменяет проверки сервисов в health-check на заглушки.
    
    - check_qdrant и check_ollama → AsyncMock (чтобы работали с await)
    - check_docs и run_rag_canary_check → синхронные заглушки, всегда возвращают True
    
    Тесты не зависят от реального состояния Qdrant/Ollama и не делают сетевые запросы.
    """
    mock_qdrant = AsyncMock(return_value=True)
    mock_ollama = AsyncMock(return_value=True)

    def _true():
        return True

    with patch('app.health.check_qdrant', mock_qdrant):
        with patch('app.health.check_ollama', mock_ollama):
            with patch('app.health.check_docs', side_effect=_true):
                with patch('app.health.run_rag_canary_check', side_effect=_true):
                    yield