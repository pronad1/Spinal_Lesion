import os
import multiprocessing

# CRITICAL: Bind to the PORT environment variable (Render requirement)
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5000))

# Worker configuration - OPTIMIZED for Render's 512MB memory limit
workers = 1  # Single worker - CRITICAL for low memory
worker_class = "sync"
threads = 1  # Single thread - prevents memory multiplication
timeout = 600  # 10 minutes - allow time for first model load
keepalive = 2
graceful_timeout = 60  # Longer grace period for cleanup

# Logging - IMPORTANT: Shows port binding in logs
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True  # Capture app output to logs

# Memory optimization - CRITICAL
max_requests = 100  # Restart worker periodically to prevent memory leaks
max_requests_jitter = 20
worker_tmp_dir = "/dev/shm"  # Use shared memory for worker files (faster, less I/O)

# Startup configuration
preload_app = False  # CHANGED: Don't preload - let lazy loading work on first request

# Limit request sizes to prevent memory spikes
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Hook to verify port binding
def on_starting(server):
    """Called before master process is initialized"""
    print(f"🚀 Starting Gunicorn on {bind}")
    print(f"📊 Workers: {workers}, Threads: {threads}")
    print(f"💾 Memory optimization: Single worker, lazy model loading")

