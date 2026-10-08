import uuid
import hashlib
import time
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance

from app.logger import logger
from app.settings import settings


# Конфигурация векторного хранилища
collection_name = settings.QDRANT_COLLECTION_NAME
embedding_model_name = settings.EMBEDDING_MODEL_NAME
vector_size = settings.VECTOR_SIZE

# ─── Функция инициализации Qdrant с retry ────────────────────
def initialize_qdrant() -> QdrantClient:
    """
    Подключается к Qdrant, создаёт коллекцию (если нет).
    Повторяет попытки до 30 раз с интервалом 2 секунды.
    """
    max_retries = 30
    interval = 2  # секунд

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f'Подключение к Qdrant (попытка {attempt}/{max_retries})...')

            client = QdrantClient(
                settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10.0  # таймаут на каждый запрос
            )

            # Проверяем, что Qdrant отвечает
            client.get_collections()
            logger.info('Qdrant отвечает')

            # Создаём коллекцию, если её нет
            if not client.collection_exists(collection_name):
                logger.info(f'Создание коллекции {collection_name}...')
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f'Коллекция {collection_name} создана')
            else:
                logger.info(f'Коллекция {collection_name} уже существует')

            return client

        except Exception as exc:
            logger.warning(f'Не удалось подключиться к Qdrant: {exc}')
            if attempt == max_retries:
                raise RuntimeError(
                    f'Не удалось подключиться к Qdrant за {max_retries} попыток'
                ) from exc
            time.sleep(interval)




# ─── Глобальная переменная для клиента (заполняется при старте) ───
client: QdrantClient | None = None
embeddings: OllamaEmbeddings | None = None
vector_store: QdrantVectorStore | None = None


def initialize_rag() -> None:
    """
    Вызывается один раз при старте приложения (в lifespan).
    Инициализирует клиент Qdrant, embedding-модель и vector store.
    """
    global client, embeddings, vector_store

    # 1. Подключаемся к Qdrant (с retry)
    client = initialize_qdrant()

    # 2. Инициальзируем embedding-модель
    logger.info(f'Загрузка embedding-модели: {embedding_model_name}...')
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    logger.info('Embedding-модель загружена')

    # 3. Создаем vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        distance=Distance.COSINE
    )
    logger.info('Vector store инициализирован')    


def initialize_rag_from_docs() -> None:
    """Загружает все .md-файлы из директории docs/ в векторную базу при старте сервиса."""
    if vector_store is None:
        logger.error('Vector store не инициализирован.')
        return
    
    docs_dir = Path('docs')
    if not docs_dir.exists():
        logger.warning('Директория docs/ не найдена')
        return

    documents = []
    doc_ids = []   # ← Храним ID параллельно с документами


    for file_path in docs_dir.glob('*.md'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Генерируем детерминированный ID из пути
                    hash_hex = hashlib.md5(str(file_path).encode()).hexdigest()
                    doc_id = str(uuid.UUID(hash_hex[:32]))
                    documents.append(
                        Document(page_content=content, metadata={'source': str(file_path)})
                    )
                    doc_ids.append(doc_id)
        except Exception as exc:
            logger.error(f'Ошибка чтения файла {file_path}: {exc}')

    if documents:
        # Передаем ids - при повторном добавлении с тем же ID документ обновится, а не продублируется
        vector_store.add_documents(documents, ids=doc_ids)
        logger.info(f'Загружено {len(documents)} документов в RAG-хранилище')
    else:
        logger.warning('В директории docs/ не найдено .md-файлов')


def add_document_to_index(file_path: str) -> bool:
    """
    Инкрементально добавляет один документ в векторную БД.
    Возвращает True при успехе.
    """
    if vector_store is None:
        logger.error('Vector store не инициализирован.')
        return False

    
    try:
        # 1. Читаем файл
        content = Path(file_path).read_text(encoding="utf-8").strip()
        if not content:
            return False

        # 2. Создаём Document с метаданными
        doc = Document(
            page_content=content,
            metadata={"source": str(file_path)}
        )

        # 3. Генерируем детерминированный ID из пути (чтобы не дублировать)
        hash_hex = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_id = str(uuid.UUID(hash_hex[:32]))

        # 4. Добавляем в Qdrant через LangChain-обёртку
        # Используем упреждающую проверку: если документ уже есть — обновляем
        vector_store.add_documents([doc], ids=[doc_id])

        logger.info(f'Документ добавлен в индекс: {file_path}')
        return True

    except Exception as exc:
        logger.error(f'Ошибка при добавлении документа {file_path}: {exc}')
        return False


def search_documentation(query: str, k: int = 1, similarity_threshold: float = 0.62) -> str | None:
    """Выполняет семантический поиск по документации API."""
    if vector_store is None:
        logger.error('Vector store не инициализирован.')
        return None

    
    try:
        logger.info(f'Семантический поиск: {query!r}')
        results = vector_store.similarity_search_with_score(
            query, k=k, score_threshold=similarity_threshold
        )

        if results:
            doc, score = results[0]
            logger.info(f'Найден релевантный документ ({doc.metadata["source"]}) (score={score:.3f}) для {query!r}')
            return doc.page_content

        logger.info(f'Релевантные документы не найдены для {query!r}')
        return None

    except Exception as exc:
        logger.error(f'Ошибка при выполнении RAG-поиска для {query!r}: {exc}', exc_info=True)
        return None