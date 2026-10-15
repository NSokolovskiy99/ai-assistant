from app.schemas import SearchRequest, SearchResponse, GenerateRequest, GenerateResponse


class TestSearchRequest:
    """Тесты модели SearchRequest."""

    def test_valid_query(self):
        """Arrange: валидная строка.
        Act: создание SearchRequest.
        Assert: query сохранён корректно."""
        # Arrange & Act
        request = SearchRequest(query='Тестовый запрос')

        # Assert
        assert request.query == 'Тестовый запрос'

    def test_empty_query(self):
        """Arrange: пустая строка.
        Act: создание SearchRequest.
        Assert: query пустой, но валидный."""
        # Arrange & Act
        request = SearchRequest(query='')

        # Assert
        assert request.query == ''


class TestSearchResponse:
    """Тесты модели SearchResponse."""

    def test_found_with_content(self):
        """Arrange: документ найден.
        Act: создание SearchResponse.
        Assert: found=True, content заполнен, message=None."""
        # Arrange & Act
        response = SearchResponse(found=True, content='### GET /api/v1/users')

        # Assert
        assert response.found is True
        assert response.content == '### GET /api/v1/users'
        assert response.message is None

    def test_not_found_with_message(self):
        """Arrange: документ не найден.
        Act: создание SearchResponse.
        Assert: found=False, message заполнен, content=None."""
        # Arrange & Act
        response = SearchResponse(found=False, message='Документация не найдена')

        # Assert
        assert response.found is False
        assert response.message == 'Документация не найдена'
        assert response.content is None


class TestGenerateRequest:
    """Тесты модели GenerateRequest."""

    def test_valid_query(self):
        """Arrange: валидная строка.
        Act: создание GenerateRequest.
        Assert: query сохранён корректно."""
        # Arrange & Act
        request = GenerateRequest(query='Создать документацию для API')

        # Assert
        assert request.query == 'Создать документацию для API'


class TestGenerateResponse:
    """Тесты модели GenerateResponse."""

    def test_success_with_content(self):
        """Arrange: успешная генерация.
        Act: создание GenerateResponse.
        Assert: success=True, content и file_path заполнены."""
        # Arrange & Act
        response = GenerateResponse(
            success=True,
            message='Документ создан',
            content='### POST /api/v1/tasks',
            file_path='docs/api_task.md'
        )

        # Assert
        assert response.success is True
        assert response.message == 'Документ создан'
        assert response.content == '### POST /api/v1/tasks'
        assert response.file_path == 'docs/api_task.md'

    def test_failure_without_content(self):
        """Arrange: неудачная генерация.
        Act: создание GenerateResponse.
        Assert: success=False, message содержит ошибку, content=None."""
        # Arrange & Act
        response = GenerateResponse(success=False, message='Ошибка генерации: неверный формат')

        # Assert
        assert response.success is False
        assert response.message == 'Ошибка генерации: неверный формат'
        assert response.content is None
        assert response.file_path is None