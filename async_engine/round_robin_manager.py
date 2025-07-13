import asyncio
from typing import Any
from contextlib import AbstractAsyncContextManager



class AsyncRoundRobin(AbstractAsyncContextManager):

    def __init__(self):
        self.queue = asyncio.Queue()
        self.resources = []

    async def __aenter__(self):
        resource = await self.queue.get()
        assert resource.in_use is False, "Resource is already in use"
        resource.in_use = True
        return resource

    def add_resource(self, data: Any = None):
        # Adds more resouces: Data is the API key
        # (same API key can be added more than once)
        resource = Resource(self, data)
        self.resources.append(resource)
        self.queue.put_nowait(resource)

    async def __aexit__(self, exc_type, exc, tb):
        return None


class Resource:

    # ToDo: add support for limiting the requests per minute (this should not be the bottleneck for now)

    def __init__(self, limiter: AsyncRoundRobin, data: Any = None):
        self.data = data
        self.in_use = False
        self.limiter = limiter

    def reschedule(self, time_taken: float, amount_used: float):
        self.limiter.queue.put_nowait(self)

    def free(self, time_taken: float, amount_used: float):
        self.reschedule(time_taken, amount_used)
        self.in_use = False


if __name__ == "__main__":
    limiter = AsyncRoundRobin()
    limiter.add_resource(data="private key 1")
    limiter.add_resource(data="private key 2")


    async def main():
        for idx in range(100):
            async with limiter as resource:
                print(f"{idx} resource acquired with data {resource.data}")
                resource.free(0.1, 6000)


    asyncio.run(main())
