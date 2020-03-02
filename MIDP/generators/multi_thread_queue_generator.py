from queue import Queue, Full, Empty
from threading import Thread
import threading
import types
import psutil


class MultiThreadQueueGenerator:

    def __init__(
        self,
        queue_size=0,
        n_workers=2,
        verbose=0,
        reserved_mem=psutil.virtual_memory().available / 10,  # 1/10
        # reserved_mem=(50 * 1024**3),  # 50GB
        timeout=100,  # 100 secs
    ):
        # init variables
        self.queue_size = queue_size
        self.n_workers = n_workers
        self.verbose = verbose
        self.timeout = timeout

        # class variables
        self.threads = []
        self.queueing = False
        self.generator = None
        self.queue = None
        self.reserved_mem = reserved_mem
        self.name = self.__class__.__name__.split('_')[-1]

    def reset_threads(self):
        self.queueing = False
        for thread in self.threads:
            thread.join()
            assert not thread.is_alive()

        if self.verbose:
            print('[%s] reset' % self.name)

        # if self.queue is not None:
        #     print('[%s] joining' % self.name)
        #     self.queue.join()
        #     print('[%s] joining done' % self.name)
        # self.queue = None
        if self.queue is not None:
            self.queue.mutex.acquire()
            self.queue.queue.clear()
            self.queue.all_tasks_done.notify_all()
            self.queue.unfinished_tasks = 0
            self.queue.mutex.release()
            self.queue.join()
            assert self.queue.qsize() == 0

    def _init_jobs(self):
        raise NotImplementedError

    # should be a function
    def _producer_work(self):
        raise NotImplementedError

    def _producer(self):
        thread_name = threading.currentThread().getName()

        def put_data_into_queue(data):
            while self.queueing:
                try:
                    if psutil.virtual_memory().available <= self.reserved_mem:
                        if self.verbose:
                            print('Memroy leak!')
                    else:
                        self.queue.put_nowait(data)
                        break

                except Full:
                    pass

            if self.verbose:
                print('[%s] %s putted, size: %d/%d' % (
                    self.name,
                    thread_name,
                    self.queue.qsize(),
                    self.queue_size
                ))

        while threading.main_thread().is_alive():
            producer_end = False
            try:
                result = self._producer_work()
            except (IndexError, StopIteration):
                producer_end = True

            if producer_end or (result is None) or not self.queueing:
                if self.verbose:
                    print('[%s] %s =done=, size: %d/%d' % (
                        self.name,
                        thread_name,
                        self.queue.qsize(),
                        self.queue_size
                    ))
                break

            if isinstance(result, types.GeneratorType):
                for sub_result in result:
                    put_data_into_queue(sub_result)
                if not self.queueing:
                    break
            else:
                put_data_into_queue(result)

    # should be a generator
    def _consumer(self):
        return self._fast_yield()

    def _fast_yield(self):
        data = None
        for _ in range(len(self)):

            # Try to retreive new data
            if data is None:
                data = self.queue.get(timeout=self.timeout)
                self.queue.task_done()

            # but use the previous data if retreiving is too slow
            else:
                try:
                    data = self.queue.get(timeout=self.timeout)
                    self.queue.task_done()
                except Empty:
                    pass

            yield data

    @property
    def done(self):
        for thread in self.threads:
            if thread.is_alive():
                return False
        return True

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):

        if self.verbose:
            print('[%s] initialized' % self.name)

        self.reset_threads()
        self._init_jobs()

        # init queue
        self.queue = Queue(maxsize=self.queue_size)

        # start queueing
        self.queueing = True

        # multi-thread producers
        for _ in range(self.n_workers):
            thread = Thread(target=self._producer)
            thread.setDaemon(True)
            thread.start()
            self.threads.append(thread)

        # single thread consumer
        return self._consumer()
