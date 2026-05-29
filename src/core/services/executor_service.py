from concurrent.futures import ThreadPoolExecutor


class ExecutorService:
    def __init__(self, max_workers=10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending = []
        self.futures = []

    def submit(self, func, *args, **kwargs):
        self.pending.append((func, args, kwargs))

    def execute(self):
        for func, args, kwargs in self.pending:
            future = self.executor.submit(func, *args, **kwargs)
            self.futures.append(future)

        self.pending.clear()

    def wait(self):
        for future in self.futures:
            future.result()

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.execute()
        self.wait()
        self.shutdown()

