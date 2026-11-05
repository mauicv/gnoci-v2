import threading
import time
import traceback

class Loop:
    def __init__(self, hz, func, limit=None):
        self.interval = 1.0 / hz
        self.func = func
        self._running = False
        self._thread = None
        self.limit = limit
        self.count = 0
        self.failed = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            next_tick = time.perf_counter()
            while self._running:
                self.func()
                self.count += 1
                if self.limit and self.count >= self.limit:
                    self._running = False
                    break
                next_tick += self.interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Behind schedule: resync rather than run missed ticks back to back.
                    next_tick = time.perf_counter()
        except Exception:
            # Can't join this thread from inside itself, so just mark it stopped.
            traceback.print_exc()
            self.failed = True
            self._running = False

    def stop(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()