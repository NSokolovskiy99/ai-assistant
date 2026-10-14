### Метод PATCH /api/v1/tasks/{id}

**Описание**: Обновляет существующую задачу.

**Параметры пути**:
- `id` (integer): идентификатор задачи

**Тело запроса**:
```json
{"title": "string", "completed": true}
```

**Ответ**:
```json
{"id": 123, "title": "string", "completed": true}
```