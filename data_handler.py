import csv
import os
import time
from collections import deque

class SquatDataCollector:
    # Setting up the data members
    def __init__(self, filename="squat_dataset.csv", delay=10.0):
        self.filename = filename
        self.delay = delay
        self.is_recording = False
        self.start_time = 0
        # stores timestamp, data_snapshot
        self.buffer = deque()

    """A function to start the collecting of data"""
    def start(self):
        self.is_recording = True
        self.start_time = time.time()
        self.buffer.clear()
        print(f"Data Collection STARTED (First {self.delay}s will be ignored)")

    """A function to stop recording (is triggered on the second r key click)"""
    def stop(self):
        self.is_recording = False
        # Clearing the buffer, the last 10s of data wont hit the CSV.
        self.buffer.clear()
        print(f"Data Collection STOPPED (Last {self.delay}s discarded)")

    """ This function is to collect data frames for training the model. It discards the first 10
        seconds and the last 10 seconds to give time to get into and out of frame
    """
    def collect(self, data_snapshot):
        # Return if not set to recording mode
        if not self.is_recording:
            return
        
        # Get the current time
        now = time.time()

        # Get the time between the start of the recording and the time now
        elapsed = now - self.start_time

        # Ignore the first 10 seconds (Start Delay)
        if elapsed < self.delay:
            return

        # Add current data to the "Waiting Room"
        self.buffer.append((now, data_snapshot))

        # If the oldest item has been waiting 10s, it's fine to write
        while self.buffer and (now - self.buffer[0][0]) >= self.delay:
            timestamp, safe_data = self.buffer.popleft()
            self._write_to_csv(safe_data)

    """A function to write the training data to the csv"""
    def _write_to_csv(self, data):
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['lean', 'asym', 'r_angle', 'l_angle', 'weight_diff', 'label'])
            writer.writerow(data)