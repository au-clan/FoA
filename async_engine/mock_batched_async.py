import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass


class BatchingAPI:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.futures = []
        self.prompts = []

        # for debugging, counts the number of batches processed so far
        self.num_batches_processed = 0

    async def buffered_request(self, prompt):
        """
        public facing API, returns a future that will be resolved with the result of the request.
        the request is buffered and only sent when a full batch of prompts has been collected
        """
        # create future for prompt
        future = asyncio.Future()

        # add coroutine to batch
        self.futures.append(future)
        self.prompts.append(prompt)

        # add a concurrent task to check resolve the batch
        asyncio.create_task(self.resolve())
        # or just await resolve directly
        # await self.resolve()

        return await future

    async def resolve(self):
        if len(self.futures) < self.batch_size:
            return

        # create a new list for the batch to be processed
        futures_to_resolve = self.futures[:self.batch_size]
        prompts_to_resolve = self.prompts[:self.batch_size]

        # clear the lists
        self.futures = self.futures[self.batch_size:]
        self.prompts = self.prompts[self.batch_size:]

        await self.flush(futures_to_resolve, prompts_to_resolve)

    async def flush(self, futures_to_resolve, prompts_to_resolve):

        # ToDo: we could add a timeout based resolve mechanism:
        # even if there's not enough samples for a full batch, we resolve after a timeout
        # this could be done with a separate task that sleeps for a while and then resolves the batch
        # if we want to do this, then these asserts probably need to go
        assert len(futures_to_resolve) == len(prompts_to_resolve)
        assert len(futures_to_resolve) == self.batch_size

        # find duplicate prompts
        prompt2futures = defaultdict(list)
        for prompt, future in zip(prompts_to_resolve, futures_to_resolve):
            prompt2futures[prompt].append(future)

        # make requests
        request_coroutines = []
        for prompt, futures in prompt2futures.items():
            request_coroutines.append(self.immediate_request(prompt, n=len(futures)))
        request_results = await asyncio.gather(*request_coroutines)

        # resolve futures
        for results, futures in zip(request_results, prompt2futures.values()):
            assert len(results) == len(futures)
            for result, future in zip(results, futures):
                #print(f"Setting result for future {future}, current state: {future.done()}")
                future.set_result(result)

    async def immediate_request(self, prompt, n):
        """
        used to process a request without buffering.
        used internally by the resolve method, can also be used publicly for one-off prompts that don't fit into the
        lock-step mechanism of N prompts in a batch
        """

        # ToDo, this is a placeholder
        # in the actual implementation, we would send a request to openai here

        results = []
        for i in range(n):
            # we're storing the number of batches so far, the index in this batch and the prompt
            # this combination should be unique, so we can check for duplicates
            # just doing this for debugging
            results.append((self.num_batches_processed, i, prompt))

        self.num_batches_processed += 1

        # sleep for a random amount of time to simulate network latency
        await asyncio.sleep(random.random() * 0.05)
        return results


@dataclass(frozen=True)
class State:
    prompt: str


class Agent:

    @staticmethod
    async def step(state: State, api: BatchingAPI):
        # make request
        result = await api.buffered_request(state.prompt)

        # do something with the result
        # ...

        return result


if __name__ == "__main__":

    # run code repeatedly, for stress testing
    for _ in range(100):
        api = BatchingAPI(batch_size=3)

        # set up prompts
        num_prompts = 30
        assert num_prompts % api.batch_size == 0
        prompts = []
        for _ in range(num_prompts):
            prompt = random.randint(0, 10)
            prompts.append(f"prompt{prompt}")
        states = [State(prompt=prompt) for prompt in prompts]


        async def main():
            coroutines = []
            for state in states:
                coroutines.append(Agent.step(state, api))
            results = await asyncio.gather(*coroutines)
            return results


        results = asyncio.run(main())

        # now we can check the results
        assert len(results) == len(prompts)

        # within each batch, the combination of prompt and batch_idx must be unique
        # this means that each combination of batch_num, prompt and batch_idx must occur exactly once
        # we can use a set to check this
        assert len(set(results)) == len(results)
