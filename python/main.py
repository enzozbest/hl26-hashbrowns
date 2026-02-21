import asyncio

from hashbrowns.config import settings
from hashbrowns.ibex.client import IbexClient

async def main():
    async with IbexClient(settings) as client:
        results = await client.search([528349, 186246], 50, 27700)
        print(results)

if __name__ == "__main__":
   asyncio.run(main())
