import csv
import os
import time
from collections import deque

class SquatDataCollector:
    def __init__(self, filename="squat_dataset.csv", delay=10.0):
        self.filename = filename
        self.delay = delay
        self.is_recording = False
        self.start_time = 0
        # The "Waiting Room": stores (timestamp, data_snapshot)
        self.buffer = deque()

    def start(self):
        self.is_recording = True
        self.start_time = time.time()
        self.buffer.clear()
        print(f"--- Data Collection STARTED (First {self.delay}s will be ignored) ---")

    def stop(self):
        self.is_recording = False
        # By clearing the buffer, the last 10s of data never hits the CSV.
        self.buffer.clear()
        print(f"--- Data Collection STOPPED (Last {self.delay}s discarded) ---")

    def collect(self, data_snapshot):
        if not self.is_recording:
            return

        now = time.time()
        elapsed = now - self.start_time

        # Ignore the first 10 seconds (Start Delay)
        if elapsed < self.delay:
            return

        # Add current data to the "Waiting Room"
        self.buffer.append((now, data_snapshot))

        # If the oldest item has been waiting 10s, it's "safe" to write
        while self.buffer and (now - self.buffer[0][0]) >= self.delay:
            timestamp, safe_data = self.buffer.popleft()
            self._write_to_csv(safe_data)

    def _write_to_csv(self, data):
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['lean', 'asym', 'r_angle', 'l_angle', 'weight_diff', 'label'])
            writer.writerow(data)