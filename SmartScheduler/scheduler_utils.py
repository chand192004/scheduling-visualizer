import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.priority = priority
        self.completion_time = 0
        self.turnaround_time = 0
        self.waiting_time = 0

def extract_features(processes):
    """Extract features from a set of processes"""
    if not processes:
        return None
        
    # Extract basic process set features
    n_proc = len(processes)
    arrival_times = [p.arrival_time for p in processes]
    burst_times = [p.burst_time for p in processes]
    priorities = [p.priority for p in processes]
    
    # Calculate the requested metrics for model training
    features = {
        'n_proc': n_proc,                                    # Number of processes
        'avg_arrival': sum(arrival_times) / n_proc,          # Average arrival time
        'burst_std': np.std(burst_times) if len(burst_times) > 1 else 0,  # Std dev of burst times
        'has_priority': 1 if any(p != 1 for p in priorities) else 0,      # Presence of priority values
        'burst_variance': np.var(burst_times) if len(burst_times) > 1 else 0,  # CPU burst variance
        'avg_burst': sum(burst_times) / n_proc,             # Average burst time
        'min_burst': min(burst_times) if burst_times else 0,  # Min burst time
        'max_burst': max(burst_times) if burst_times else 0,  # Max burst time
        'max_waiting': max([p.waiting_time for p in processes]) if processes else 0,  # Max waiting time
        'time_quantum': 2  # Default time quantum for RR
    }
    
    return features

def average_metrics(processes):
    """Calculate average turnaround time and waiting time"""
    n = len(processes)
    total_tat = sum(p.turnaround_time for p in processes)
    total_wt = sum(p.waiting_time for p in processes)
    avg_tat = total_tat / n
    avg_wt = total_wt / n
    return avg_tat, avg_wt

def draw_gantt_chart(gantt, title):
    """Draw Gantt chart for process scheduling"""
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    
    for i, (pid, start, end) in enumerate(gantt):
        ax.broken_barh([(start, end - start)], (10, 9), 
                      facecolors=colors[pid % len(colors)])
        ax.text((start + end) / 2, 14, f"P{pid}", 
               ha='center', va='center', color='black')
    
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_title(f"Gantt Chart: {title}")
    ax.grid(True)
    plt.tight_layout()
    plt.show() 