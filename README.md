# Лабораторная работа №2 — REST API (FastAPI)

Распределённая система бенчмаркинга моделей оценки глубины (Depth Estimation).
Реализация REST API сервиса взаимодействия с базой данных по протоколу HTTP с архитектурой **Repository + Service Layer**, строгим разделением **DTO (Pydantic)** и **ORM (SQLAlchemy 2.0 asyncpg)**, а также аутентификацией на базе **JWT**.

---

## Архитектура

Запросы проходят через изолированные слои:

```
[ HTTP-запрос (JSON) ]
          │
          ▼
   1. [ API Router ] (FastAPI)
      - Приём Request DTO (валидация входных данных через Pydantic)
      - Dependency Injection (`Depends`): сессии БД, аутентификация, сервисы
      - Возврат Response DTO и HTTP-статусов (200, 201, 204, 400, 401, 403, 404, 409)
          │
          ▼
   2. [ Service Layer ] (Бизнес-логика)
      - Хеширование паролей (bcrypt), генерация и проверка JWT (PyJWT)
      - Бизнес-правила: уникальность email/имен моделей, валидация активности моделей
      - Оркестрация репозиториев
          │
          ▼
   3. [ Repository Layer ] (Доступ к данным)
      - Инкапсуляция SQL/ORM запросов (`select`, `selectinload`, `flush`, `delete`)
      - Работа с `AsyncSession` и моделями SQLAlchemy
          │
          ▼
   4. [ Database ] (PostgreSQL 16)
```

---

## Основные эндпоинты

### Аутентификация (`/api/v1/auth`)
- `POST /api/v1/auth/register` — регистрация нового пользователя (пароль хешируется через `bcrypt`, возвращается DTO без хеша пароля).
- `POST /api/v1/auth/login` — аутентификация, выдача `access_token` (JWT).
- `GET /api/v1/auth/me` — получение профиля текущего авторизованного пользователя (`Authorization: Bearer <token>`).

### Нейросети (`/api/v1/models`)
- `GET /api/v1/models` — список моделей с пагинацией и фильтром по `is_active`.
- `GET /api/v1/models/{id}` — получение модели по ID.
- `POST /api/v1/models` — добавление новой модели (требуется авторизация).
- `PATCH /api/v1/models/{id}` — частичное обновление модели.
- `DELETE /api/v1/models/{id}` — удаление модели (ролевая модель: доступно только роли `ADMIN`).

### Датасеты (`/api/v1/datasets`)
- `GET /api/v1/datasets` — список доступных датасетов.
- `GET /api/v1/datasets/{id}` — датасет по ID.
- `POST /api/v1/datasets` — регистрация датасета (ролевая модель: доступно ролям `OPERATOR` и `ADMIN`).

### Бенчмарки и связь многие-ко-многим (`/api/v1/experiments`)
- `POST /api/v1/experiments` — создание эксперимента со списком моделей (`model_ids: [1, 2, ...]`). Сервис валидирует активность моделей и формирует связанные записи запусков `ExperimentModelRun` (M:N).
- `GET /api/v1/experiments` — список экспериментов.
- `GET /api/v1/experiments/{id}` — детальная информация об эксперименте со всеми включёнными моделями и их метриками.
- `GET /api/v1/experiments/{id}/leaderboard` — получение ранжированного лидерборда моделей по метрикам (`rmse`, `silog`, `absrel`, `delta_1`, `time`).

---

## Запуск приложения

1. Поднять базу данных PostgreSQL:
```bash
docker compose up -d
```

2. Применить миграции схемы БД:
```bash
alembic upgrade head
```

3. Запустить API-сервер:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

4. Документация Swagger UI доступна по адресу:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Демонстрация сценариев (cURL)

Для автоматизированного сквозного прогона всех сценариев (регистрация, RBAC, выпуск JWT, создание датасетов, моделей, экспериментов M:N, получение лидерборда и удаление):

```bash
./scripts/demo_api.sh
```

---

## Тестирование

Запуск модульных и интеграционных тестов (репозитории + HTTP API эндпоинты):

```bash
pytest -v
```

Проверка соответствия моделей и миграций БД:
```bash
alembic check
```
