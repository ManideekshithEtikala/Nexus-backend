import sys
import os

# This adds the root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database.database import engine, init_db
from app.models.model import Base  # import your models so tables register
import asyncio

async def test_connection():
    try:
        async with engine.connect() as conn:
            print("✅ Database connected successfully!")
        await init_db()
        print("✅ Tables created successfully!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test_connection())