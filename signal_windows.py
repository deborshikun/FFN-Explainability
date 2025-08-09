import threading
import time
import sys

# Define SIGALRM directly in the module
SIGALRM = 14

class Timer:
    def __init__(self):
        self.handler = None
        self._timer = None

    def _raise_alarm(self):
        if self.handler is not None:
            try:
                self.handler(SIGALRM, None)
            except Exception:
                # Gracefully exit on timeout
                sys.exit(0)

def signal(signum, handler):
    """Simulate signal registration"""
    global _timer
    if signum == SIGALRM:
        if _timer is None:
            _timer = Timer()
        _timer.handler = handler

def alarm(seconds):
    """Set an alarm"""
    global _timer
    if _timer is None:
        return
    
    if _timer._timer is not None:
        _timer._timer.cancel()
    
    _timer._timer = threading.Timer(seconds, _timer._raise_alarm)
    _timer._timer.daemon = True  # Make timer thread daemon
    _timer._timer.start()
    return 0

# Create global timer instance
_timer = Timer()