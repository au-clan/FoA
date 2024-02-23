class BatchingAPI:

    async def request(self, prompt):
        # create coroutine for prompt
        coroutine = ...

        # add coroutine to batch
        self.coroutines.append(coroutine)
        self.prompts.append(prompt)

        # add a concurrent task to check resolve the batch
        ...

        return coroutine

    async def resolve(self):
        if len(self.batch)<N:
            return

        # go through all prompts, find duplicates and join them
        # send them to API
        # await results

        # match results with coroutines in self.batch
        # resolve them


class Consumer:

    async def step(self, state, api: BatchingAPI):

        # construct prompt from state
        prompt = ...

        # make request
        result = await api.request(prompt)

        # do something with the result