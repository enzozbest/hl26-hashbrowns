import asyncio
import json
from datetime import date, datetime

from hashbrowns.config import settings
from hashbrowns.ibex.client import IbexClient

def default_encoder(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

async def main():
    async with IbexClient(settings) as client:
        results = await client.search([528349, 186246], radius=50, srid=27700)
        items = results if isinstance(results, list) else results.get("items", results)
        for item in items[:10]:
            print(json.dumps(item.model_dump(exclude={"geometry"}), indent=2, default=default_encoder))

if __name__ == "__main__":
   asyncio.run(main())
