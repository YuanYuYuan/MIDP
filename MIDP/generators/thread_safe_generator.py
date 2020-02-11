import threading


class ThreadSafeGenerator:

    def __init__(self, gen):
        self.gen = gen
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.gen)


def thread_safe_gen(f):
    def g(*a, **kw):
        return ThreadSafeGenerator(f(*a, **kw))
    return g
