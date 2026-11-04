#!/usr/bin/env python3
"""Скрипт ручного или автоматического запуска инициализации БД начальными данными."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.seed import seed_initial_data
from src.database import AsyncSessionLocal


async def main():
    print("Инициализация начальных данных БД...")
    async with AsyncSessionLocal() as session:
        await seed_initial_data(session)
    print("Инициализация успешно завершена!")


if __name__ == "__main__":
    asyncio.run(main())
