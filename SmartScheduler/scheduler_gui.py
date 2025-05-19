import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import random
from scheduler_utils import Process, extract_features, average_metrics, draw_gantt_chart
from ml_predictor import SchedulerPredictor
from Smart_scheduler import fcfs, sjf, round_robin, priority_scheduling
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SchedulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Scheduler - ML Based")
        self.processes = []
        self.predictor = SchedulerPredictor()
        self.dataset_path = "scheduler_dataset.csv"
        self.setup_ui()

    def setup_ui(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="Add Process")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(input_frame, text="Arrival Time:").grid(row=0, column=0)
        self.arrival_var = tk.IntVar()
        ttk.Entry(input_frame, textvariable=self.arrival_var, width=5).grid(row=0, column=1)

        ttk.Label(input_frame, text="Burst Time:").grid(row=0, column=2)
        self.burst_var = tk.IntVar()
        ttk.Entry(input_frame, textvariable=self.burst_var, width=5).grid(row=0, column=3)

        ttk.Label(input_frame, text="Priority:").grid(row=0, column=4)
        self.priority_var = tk.IntVar()
        ttk.Entry(input_frame, textvariable=self.priority_var, width=5).grid(row=0, column=5)

        ttk.Button(input_frame, text="Add", command=self.add_process).grid(row=0, column=6, padx=5)
        ttk.Button(input_frame, text="Clear All", command=self.clear_all).grid(row=0, column=7, padx=5)

        # Dataset frame
        dataset_frame = ttk.LabelFrame(self.root, text="Dataset Testing")
        dataset_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Button(dataset_frame, text="Load from Dataset", command=self.load_from_dataset).grid(row=0, column=0, padx=5)
        ttk.Button(dataset_frame, text="Run All Algorithms", command=self.run_all_algorithms).grid(row=0, column=1, padx=5)

        # Process list
        self.tree = ttk.Treeview(self.root, columns=("PID", "Arrival", "Burst", "Priority"), show="headings", height=6)
        self.tree.heading("PID", text="PID")
        self.tree.heading("Arrival", text="Arrival Time")
        self.tree.heading("Burst", text="Burst Time")
        self.tree.heading("Priority", text="Priority")
        self.tree.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # Predict button
        ttk.Button(self.root, text="Predict Best Algorithm", command=self.predict).grid(row=3, column=0, pady=10)

        # Results
        self.result_text = tk.Text(self.root, height=12, width=60, state="disabled")
        self.result_text.grid(row=4, column=0, padx=10, pady=5)

    def show_gantt_charts(self, results):
        """Show Gantt charts for all algorithms in a new window"""
        gantt_window = tk.Toplevel(self.root)
        gantt_window.title("Gantt Charts")
        gantt_window.geometry("800x600")

        # Create a canvas with scrollbar
        canvas = tk.Canvas(gantt_window)
        scrollbar = ttk.Scrollbar(gantt_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add Gantt charts
        for i, (algo_name, result) in enumerate(results.items()):
            frame = ttk.LabelFrame(scrollable_frame, text=f"{algo_name} Gantt Chart")
            frame.pack(padx=10, pady=5, fill="x")

            # Create figure for this algorithm
            fig = plt.Figure(figsize=(8, 2))
            ax = fig.add_subplot(111)
            
            # Draw Gantt chart
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
            for j, (pid, start, end) in enumerate(result['gantt']):
                ax.broken_barh([(start, end - start)], (10, 9), 
                             facecolors=colors[pid % len(colors)])
                ax.text((start + end) / 2, 14, f"P{pid}", 
                       ha='center', va='center', color='black')
            
            ax.set_xlabel("Time")
            ax.set_yticks([])
            ax.grid(True)
            
            # Add metrics
            metrics_text = f"Avg TAT: {result['avg_tat']:.2f}, Avg WT: {result['avg_wt']:.2f}"
            ax.set_title(metrics_text)
            
            # Add to frame
            canvas_widget = FigureCanvasTkAgg(fig, frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="x", padx=5, pady=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_process(self):
        at = self.arrival_var.get()
        bt = self.burst_var.get()
        pr = self.priority_var.get()
        if bt <= 0:
            messagebox.showerror("Error", "Burst time must be > 0")
            return
        pid = len(self.processes) + 1
        proc = Process(pid, at, bt, pr)
        self.processes.append(proc)
        self.tree.insert("", "end", values=(pid, at, bt, pr))
        self.arrival_var.set(0)
        self.burst_var.set(0)
        self.priority_var.set(0)

    def clear_all(self):
        self.processes.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")

    def load_from_dataset(self):
        try:
            df = pd.read_csv("scheduler_dataset.csv")
            sample = df.sample(1).iloc[0]
            
            # Clear existing processes
            self.clear_all()
            
            # Generate processes based on dataset sample
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
                proc = Process(i + 1, at, bt, pr)
                self.processes.append(proc)
                self.tree.insert("", "end", values=(i + 1, at, bt, pr))
            
            messagebox.showinfo("Success", f"Loaded {num_procs} processes from dataset")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def run_all_algorithms(self):
        if not self.processes:
            messagebox.showwarning("Warning", "Add at least one process.")
            return
            
        algos = {
            "FCFS": fcfs,
            "SJF": sjf,
            "RR": lambda p: round_robin(p, quantum=2),
            "Priority": priority_scheduling
        }
        
        results = {}
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Algorithm Performance:\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        
        for name, algo in algos.items():
            procs = self.processes.copy()
            scheduled, gantt = algo(procs)
            avg_tat, avg_wt = average_metrics(scheduled)
            results[name] = {
                'avg_tat': avg_tat,
                'avg_wt': avg_wt,
                'total': avg_tat + avg_wt,
                'gantt': gantt
            }
            
            self.result_text.insert(tk.END, f"{name}:\n")
            self.result_text.insert(tk.END, f"  Avg Turnaround Time: {avg_tat:.2f}\n")
            self.result_text.insert(tk.END, f"  Avg Waiting Time: {avg_wt:.2f}\n")
            self.result_text.insert(tk.END, f"  Total Metric: {avg_tat + avg_wt:.2f}\n\n")
        
        # Find best algorithm
        best_algo = min(results.items(), key=lambda x: x[1]['total'])[0]
        self.result_text.insert(tk.END, f"Best Algorithm (based on metrics): {best_algo}\n")
        
        # Compare with ML prediction
        ml_result = self.predictor.predict_best_algorithm(self.processes)
        self.result_text.insert(tk.END, f"\nML Prediction: {ml_result['best_algorithm']}\n")
        
        if best_algo == ml_result['best_algorithm']:
            self.result_text.insert(tk.END, "\n✅ ML prediction matches the best algorithm!\n")
        else:
            self.result_text.insert(tk.END, "\n❌ ML prediction differs from the best algorithm.\n")
            self.result_text.insert(tk.END, "This could be due to:\n")
            self.result_text.insert(tk.END, "1. Different optimization criteria\n")
            self.result_text.insert(tk.END, "2. Training data characteristics\n")
            self.result_text.insert(tk.END, "3. Process set characteristics\n")
        
        self.result_text.config(state="disabled")
        
        # Show Gantt charts
        self.show_gantt_charts(results)

    def predict(self):
        if not self.processes:
            messagebox.showwarning("Warning", "Add at least one process.")
            return
        result = self.predictor.predict_best_algorithm(self.processes)
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Best Algorithm: {result['best_algorithm']}\n\n")
        self.result_text.insert(tk.END, "Confidence Scores:\n")
        for algo, score in result['confidence_scores'].items():
            self.result_text.insert(tk.END, f"{algo:10}: {score:.2%}\n")
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerApp(root)
    root.mainloop() 