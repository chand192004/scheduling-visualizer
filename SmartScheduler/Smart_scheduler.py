import time
import random
import pandas as pd
import copy
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from ml_predictor import SchedulerPredictor

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

def draw_gantt_chart(gantt, title):
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

def fcfs(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = 0
    gantt = []
    for p in processes:
        start = max(current_time, p.arrival_time)
        end = start + p.burst_time
        p.completion_time = end
        p.turnaround_time = end - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        current_time = end
        gantt.append((p.pid, start, end))
    return processes, gantt

def sjf(processes):
    n = len(processes)
    current_time = 0
    completed = 0
    is_done = [False] * n
    gantt = []
    while completed != n:
        idx = -1
        min_bt = float('inf')
        for i in range(n):
            if processes[i].arrival_time <= current_time and not is_done[i]:
                if processes[i].burst_time < min_bt:
                    min_bt = processes[i].burst_time
                    idx = i
        if idx == -1:
            current_time += 1
            continue
        p = processes[idx]
        start = current_time
        end = start + p.burst_time
        p.completion_time = end
        p.turnaround_time = end - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        current_time = end
        is_done[idx] = True
        completed += 1
        gantt.append((p.pid, start, end))
    return processes, gantt

def round_robin(processes, quantum=2):
    current_time = 0
    q = deque()
    i = 0
    gantt = []
    processes.sort(key=lambda x: x.arrival_time)
    while i < len(processes) or q:
        while i < len(processes) and processes[i].arrival_time <= current_time:
            q.append(processes[i])
            i += 1
        if not q:
            current_time += 1
            continue
        p = q.popleft()
        exec_time = min(quantum, p.remaining_time)
        start = current_time
        current_time += exec_time
        p.remaining_time -= exec_time
        gantt.append((p.pid, start, current_time))
        while i < len(processes) and processes[i].arrival_time <= current_time:
            q.append(processes[i])
            i += 1
        if p.remaining_time > 0:
            q.append(p)
        else:
            p.completion_time = current_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
    return processes, gantt

def priority_scheduling(processes):
    n = len(processes)
    current_time = 0
    completed = 0
    is_done = [False] * n
    gantt = []
    while completed != n:
        idx = -1
        highest = float('inf')
        for i in range(n):
            if processes[i].arrival_time <= current_time and not is_done[i]:
                if processes[i].priority < highest:
                    highest = processes[i].priority
                    idx = i
        if idx == -1:
            current_time += 1
            continue
        p = processes[idx]
        start = current_time
        end = start + p.burst_time
        current_time = end
        p.completion_time = end
        p.turnaround_time = end - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        is_done[idx] = True
        completed += 1
        gantt.append((p.pid, start, end))
    return processes, gantt

def average_metrics(processes):
    n = len(processes)
    total_tat = sum(p.turnaround_time for p in processes)
    total_wt = sum(p.waiting_time for p in processes)
    avg_tat = total_tat / n
    avg_wt = total_wt / n
    return avg_tat, avg_wt

def extract_features(processes):
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

if __name__ == '__main__':
    print("Smart Scheduler - Process Scheduling System")
    print("=" * 50)
    print("Choose input mode:")
    print("1. Manual input")
    print("2. Load from dataset")
    mode = input("Enter 1 or 2: ")

    original_processes = []

    if mode == "1":
        n = int(input("Enter number of processes: "))
        for i in range(n):
            at, bt, pr = map(int, input(f"Enter details for Process {i + 1} (arrival burst priority): ").split())
            original_processes.append(Process(i + 1, at, bt, pr))

    elif mode == "2":
        try:
            df = pd.read_csv("scheduler_dataset.csv")
            sample = df.sample(1).iloc[0]
            num_procs = int(sample['num_processes'])
            avg_burst = sample['avg_burst']
            std_burst = sample['std_burst']
            avg_priority = sample['avg_priority']
            std_priority = sample['std_priority']
            arrival_skew = sample['arrival_skew']
            
            for i in range(num_procs):
                bt = max(1, int(random.gauss(avg_burst, std_burst)))
                pr = max(1, min(5, int(random.gauss(avg_priority, std_priority))))
                at = int(random.uniform(0, arrival_skew))
                original_processes.append(Process(i + 1, at, bt, pr))
                
            print(f"\nLoaded {num_procs} processes from dataset:")
            print("PID  Arrival  Burst  Priority")
            print("-" * 30)
            for p in original_processes:
                print(f"{p.pid:<5}{p.arrival_time:<8}{p.burst_time:<6}{p.priority:<9}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            exit()
    else:
        print("Invalid input.")
        exit()

    # Extract features
    features = extract_features(original_processes)
    print("\nProcess Set Features:")
    print("-" * 50)
    for name, value in features.items():
        print(f"{name:20}: {value:.3f}")

    # Get ML prediction
    try:
        predictor = SchedulerPredictor()
        ml_result = predictor.predict_best_algorithm(original_processes)
        print("\nML Prediction Results:")
        print("-" * 50)
        print(f"Best Algorithm: {ml_result['best_algorithm']}")
        print("\nConfidence Scores:")
        for algo, score in ml_result['confidence_scores'].items():
            print(f"{algo:10}: {score:.2%}")
    except Exception as e:
        print(f"\nML prediction not available: {str(e)}")
        print("Running all algorithms for comparison...")
        ml_result = None

    # Run all algorithms and compare
    algos = {
        "FCFS": fcfs,
        "SJF": sjf,
        "RR": lambda p: round_robin(p, quantum=2),
        "Priority": priority_scheduling
    }

    print("\nRunning all algorithms for comparison...")
    results = {}
    
    for name, algo in algos.items():
        print(f"\nExecuting {name}...")
        start_time = time.time()
        procs = copy.deepcopy(original_processes)
        scheduled, gantt = algo(procs)
        exec_time = time.time() - start_time
        
        avg_tat, avg_wt = average_metrics(scheduled)
        results[name] = {
            'avg_tat': avg_tat,
            'avg_wt': avg_wt,
            'exec_time': exec_time,
            'gantt': gantt
        }
        
        print(f"Average Turnaround Time: {avg_tat:.2f}")
        print(f"Average Waiting Time  : {avg_wt:.2f}")
        print(f"Execution Time       : {exec_time:.4f} seconds")
        
        # Draw Gantt chart
        draw_gantt_chart(gantt, name)

    # Find best algorithm
    best_algo = min(results.items(), key=lambda x: x[1]['avg_tat'] + x[1]['avg_wt'])[0]
    print(f"\nBest Algorithm (based on metrics): {best_algo}")
    print("Based on combined average turnaround time and waiting time")
    
    if ml_result:
        print(f"\nML Prediction: {ml_result['best_algorithm']}")
        if best_algo == ml_result['best_algorithm']:
            print("ML prediction matches the best algorithm!")
        else:
            print("ML prediction differs from the best algorithm.")
            print("This could be due to:")
            print("1. Different optimization criteria")
            print("2. Training data characteristics")
            print("3. Process set characteristics") 