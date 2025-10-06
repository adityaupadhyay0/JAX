import threading

class autocast(threading.local):
    def __init__(self, enabled=True, dtype='float16'):
        super().__init__()
        self.enabled = enabled
        self.dtype = dtype
        self.prev = False

    def __enter__(self):
        self.prev = autocast.is_enabled()
        _autocast_stack.get_local().enabled = self.enabled
        _autocast_stack.get_local().dtype = self.dtype

    def __exit__(self, exc_type, exc_val, exc_tb):
        _autocast_stack.get_local().enabled = self.prev
        _autocast_stack.get_local().dtype = 'float16' # Reset to default

    @staticmethod
    def is_enabled():
        return _autocast_stack.get_local().enabled

    @staticmethod
    def get_autocast_dtype():
        return _autocast_stack.get_local().dtype

class _AutocastStack(threading.local):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.dtype = 'float16'

    def get_local(self):
        if not hasattr(self, 'enabled'):
            self.enabled = False
            self.dtype = 'float16'
        return self

_autocast_stack = _AutocastStack()