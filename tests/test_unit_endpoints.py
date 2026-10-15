from unittest.mock import patch


class TestHealthCheck:
    """Тесты эндпоинта /health."""

    def test_health_check_returns_200(self, client, mock_health_services):
        """Arrange: mock_health_services замаскировал Qdrant/Ollama/Docs/RAG.
        Act: GET /health.
        Assert: status 200, status='healthy', checks присутствуют."""
        # Act
        response = client.get('/health')

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'checks' in data

    def test_health_check_checks_structure(self, client, mock_health_services):
        """Arrange: mock_health_services.
        Act: GET /health.
        Assert: checks содержит все ключи со значениями True."""
        # Act
        response = client.get('/health')
        data = response.json()

        # Assert
        checks = data['checks']
        assert checks['qdrant'] is True
        assert checks['ollama'] is True
        assert checks['docs'] is True
        assert checks['rag_canary'] is True


class TestSearchDocs:
    """Тесты эндпоинта POST /search."""

    def test_search_existing_document(self, client, mock_rag):
        """Arrange: mock_rag -> возвращает документ.
        Act: POST /search.
        Assert: found=True, content не пустой."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = '### POST /api/v1/tasks\n**Описание**: Создание задачи'

        # Act
        response = client.post('/search', json={'query': 'создание задачи'})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['found'] is True
        assert data['content']

    def test_search_not_found_returns_message(self, client, mock_rag):
        """Arrange: mock_rag -> None.
        Act: POST /search.
        Assert: found=False, message содержит подсказку."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = None

        # Act
        response = client.post('/search', json={'query': 'несуществующий запрос xyz123'})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['found'] is False
        assert 'Документация не найдена' in data['message']

    def test_search_request_schema_validation(self, client):
        """Arrange: пустой body.
        Act: POST /search.
        Assert: статус 422 (validation error)."""
        # Act
        response = client.post('/search', json={})

        # Assert
        assert response.status_code == 422

    def test_search_with_empty_query(self, client, mock_rag):
        """Arrange: mock_rag -> None, пустая строка.
        Act: POST /search.
        Assert: статус 200, found в ответе."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = None

        # Act
        response = client.post('/search', json={'query': ''})

        # Assert
        assert response.status_code == 200
        assert 'found' in response.json()


class TestGenerateDocs:
    """Тесты эндпоинта POST /generate."""

    def test_generate_new_document(self, client, mock_rag, mock_agents, mock_storage):
        """Arrange: mock_rag -> None, mock_agents -> валидный контент, mock_storage -> путь.
        Act: POST /generate.
        Assert: success=True, content начинается с '###', file_path возвращён."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = None

        # Act
        response = client.post('/generate', json={'query': 'Поиск по ключевым словам API'})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['content'].startswith('###')
        assert data['file_path'] == 'docs/test_doc.md'

    def test_generate_duplicate_document(self, client, mock_rag):
        """Arrange: mock_rag -> найден существующий документ.
        Act: POST /generate.
        Assert: success=False, message подсказывает использовать /search."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = 'Существующий документ'

        # Act
        response = client.post('/generate', json={'query': 'создание задачи'})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is False
        assert 'Документ уже существует' in data['message']

    def test_generate_invalid_format(self, client, mock_rag):
        """Arrange: mock_rag -> None, генерация без '###'.
        Act: POST /generate.
        Assert: success=False, ошибка формата."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = None

        with patch('app.main.generate_and_validate_documentation', return_value='Без заголовка ###'):
            # Act
            response = client.post('/generate', json={'query': 'неверный формат'})

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is False
            assert 'верный формат' in data['message']

    def test_generate_exception_handling(self, client, mock_rag):
        """Arrange: mock_rag -> None, генерация выбрасывает исключение.
        Act: POST /generate.
        Assert: success=False, message содержит текст ошибки."""
        # Arrange
        mock_search, _ = mock_rag
        mock_search.return_value = None

        with patch('app.main.generate_and_validate_documentation', side_effect=Exception('Ollama timeout')):
            # Act
            response = client.post('/generate', json={'query': 'тест исключения'})

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is False
            assert 'Ollama timeout' in data['message']

    def test_generate_request_schema_validation(self, client):
        """Arrange: пустой body.
        Act: POST /generate.
        Assert: статус 422 (validation error)."""
        # Act
        response = client.post('/generate', json={})

        # Assert
        assert response.status_code == 422