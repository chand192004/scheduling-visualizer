from process import Process
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class Scheduler:
    def __init__(self):
        self.processes: List[Process] = []
        self.timeline = []
        self.current_time = 0
        
    def add_process(self, process: Process):
        self.processes.append(process)
        
    def reset(self):
        self.processes = []
        self.timeline = []
        self.current_time = 0
        
    def fcfs(self):
        """First Come First Serve Scheduling"""
        self.timeline = []
        self.current_time = 0
        ready_queue = sorted(self.processes, key=lambda x: x.arrival_time)
        
        for process in ready_queue:
            if self.current_time < process.arrival_time:
                self.current_time = process.arrival_time
                
            process.start_time = self.current_time
            process.waiting_time = self.current_time - process.arrival_time
            self.current_time += process.burst_time
            process.completion_time = self.current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            
            self.timeline.append((process.pid, process.start_time, process.completion_time))
            
        return self.timeline
    
    def sjf(self):
        """Shortest Job First Scheduling"""
        self.timeline = []
        self.current_time = 0
        ready_queue = []
        completed = []
        
        while len(completed) < len(self.processes):
            # Add processes that have arrived to ready queue
            for process in self.processes:
                if process.arrival_time <= self.current_time and process not in ready_queue and process not in completed:
                    ready_queue.append(process)
            
            if not ready_queue:
                self.current_time += 1
                continue
                
            # Sort by burst time
            ready_queue.sort(key=lambda x: x.burst_time)
            current_process = ready_queue.pop(0)
            
            current_process.start_time = self.current_time
            current_process.waiting_time = self.current_time - current_process.arrival_time
            self.current_time += current_process.burst_time
            current_process.completion_time = self.current_time
            current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
            
            self.timeline.append((current_process.pid, current_process.start_time, current_process.completion_time))
            completed.append(current_process)
            
        return self.timeline
    
    def priority(self):
        """Priority Scheduling"""
        self.timeline = []
        self.current_time = 0
        ready_queue = []
        completed = []
        
        while len(completed) < len(self.processes):
            # Add processes that have arrived to ready queue
            for process in self.processes:
                if process.arrival_time <= self.current_time and process not in ready_queue and process not in completed:
                    ready_queue.append(process)
            
            if not ready_queue:
                self.current_time += 1
                continue
                
            # Sort by priority (lower number = higher priority)
            ready_queue.sort(key=lambda x: x.priority)
            current_process = ready_queue.pop(0)
            
            current_process.start_time = self.current_time
            current_process.waiting_time = self.current_time - current_process.arrival_time
            self.current_time += current_process.burst_time
            current_process.completion_time = self.current_time
            current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
            
            self.timeline.append((current_process.pid, current_process.start_time, current_process.completion_time))
            completed.append(current_process)
            
        return self.timeline
    
    def round_robin(self, quantum=2):
        """Round Robin Scheduling"""
        self.timeline = []
        self.current_time = 0
        ready_queue = []
        completed = []
        
        # Initialize remaining time for all processes
        for process in self.processes:
            process.remaining_time = process.burst_time
            
        while len(completed) < len(self.processes):
            # Add processes that have arrived to ready queue
            for process in self.processes:
                if process.arrival_time <= self.current_time and process not in ready_queue and process not in completed:
                    ready_queue.append(process)
            
            if not ready_queue:
                self.current_time += 1
                continue
                
            current_process = ready_queue.pop(0)
            
            if current_process.start_time == -1:
                current_process.start_time = self.current_time
                
            execution_time = min(quantum, current_process.remaining_time)
            self.timeline.append((current_process.pid, self.current_time, self.current_time + execution_time))
            
            self.current_time += execution_time
            current_process.remaining_time -= execution_time
            
            if current_process.remaining_time == 0:
                current_process.completion_time = self.current_time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                completed.append(current_process)
            else:
                ready_queue.append(current_process)
                
        return self.timeline
    
    def calculate_metrics(self):
        """Calculate average waiting time and turnaround time"""
        total_waiting_time = sum(process.waiting_time for process in self.processes)
        total_turnaround_time = sum(process.turnaround_time for process in self.processes)
        
        avg_waiting_time = total_waiting_time / len(self.processes)
        avg_turnaround_time = total_turnaround_time / len(self.processes)
        
        return avg_waiting_time, avg_turnaround_time 