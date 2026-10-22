# Лабораторная работа №1 — SQLAlchemy

Слой доступа к данным на асинхронном SQLAlchemy 2.0 (`asyncpg`).

# Лабораторная работа №2 — Alembic

Миграции схемы БД: вместо ручного `Base.metadata.create_all` схема описывается
версионированными миграциями Alembic в `migrations/versions/`.

## Запуск

1. Поднять базу данных:
```bash
docker compose up -d
```

2. Применить миграции (создать все таблицы):
```bash
alembic upgrade head
```

3. Запустить тесты:
```bash
pytest
```

## Работа с миграциями

Схема хранится как версия в `alembic_version` — применяется и откатывается
последовательно:

```bash
# применить все непройденные миграции
alembic upgrade head

# сгенерировать новую миграцию по изменениям моделей
alembic revision --autogenerate -m "describe change"

# откатить последнюю миграцию
alembic downgrade -1

# проверить, что модели соответствуют схеме БД
alembic check
```

Конфигурация: `alembic.ini` + `migrations/env.py` (async-шаблон, URL и
`Base.metadata` берутся из `src/database.py`).
