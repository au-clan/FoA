import asyncio
from collections import defaultdict

class BatchingAPI:
    def __init__(self, api, limiter, batch_size, timeout=30):
        # Actual API
        self.api = api
        self.limiter = limiter

        # Batching
        self.batch_size = batch_size
        self.futures = []
        self.prompts = []

        # Timeout
        self.timeout = timeout

        # for debugging, counts the number of batches processed so far
        self.num_batches_processed = 0

    async def buffered_request(self, prompt, key, namespace):
        """
        public facing API, returns a future that will be resolved with the result of the request.
        the request is buffered and only sent when a full batch of prompts has been collected
        """
        # create future for prompt
        future = asyncio.Future()

        # add coroutine to batch
        self.futures.append((future, key, namespace))
        self.prompts.append(prompt)

        # process batch if full
        await self.process_batched()

        return await future

    async def process_batched(self):
        while len(self.futures) >= self.batch_size:

            # create a new list for the batch to be processed
            futures_to_resolve = self.futures[:self.batch_size]
            prompts_to_resolve = self.prompts[:self.batch_size]

            # clear the lists
            self.futures = self.futures[self.batch_size:]
            self.prompts = self.prompts[self.batch_size:]

            await self.flush(futures_to_resolve, prompts_to_resolve)

    async def process_all(self):
        futures_to_resolve = self.futures
        prompts_to_resolve = self.prompts
        self.futures = []
        self.prompts = []

        await self.flush(futures_to_resolve, prompts_to_resolve)

    async def flush(self, futures_to_resolve, prompts_to_resolve):

        # find duplicate prompts
        prompt2futures = defaultdict(list)
        for prompt, future in zip(prompts_to_resolve, futures_to_resolve):
            prompt2futures[prompt].append(future)

        # make requests
        request_coroutines = []
        for prompt, futures in prompt2futures.items():
            request_coroutines.append(self.immediate_request(prompt, namespaces=[future[2] for future in futures]))
        request_results = await asyncio.gather(*request_coroutines)

        # resolve futures
        for results, futures in zip(request_results, prompt2futures.values()):
            futures = sorted(futures, key=lambda x: x[2])
            assert len(results) == len(futures), f"Results: {len(results)}, Futures: {len(futures)}"
            for result, future in zip(results, futures):
                #print(f"Setting result for future {future}, current state: {future.done()}")
                future[0].set_result(result)

    async def immediate_request(self, prompt, namespaces):
        """
        used to process a request without buffering.
        used internally by the resolve method, can also be used publicly for one-off prompts that don't fit into the
        lock-step mechanism of N prompts in a batch
        """

        return await self.api.request(prompt, namespaces, self.limiter)
    
    def cost(self, verbose=False)-> float:
        """
        Returns the total cost of the API calls, so far.
        """
        return self.api.cost(verbose)