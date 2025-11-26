import time
from contextlib import contextmanager

@contextmanager
def time_block(label: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f'{label}: {(end - start) * 1000:.2f} ms')
