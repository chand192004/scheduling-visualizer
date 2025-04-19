from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QTableWidget, QTableWidgetItem, QMessageBox)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scheduler import Scheduler
from process import Process

class ProcessSchedulerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scheduler = Scheduler()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Process Scheduler Visualization')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Input section
        input_layout = QHBoxLayout()
        
        # Process input fields
        self.pid_input = QLineEdit()
        self.arrival_input = QLineEdit()
        self.burst_input = QLineEdit()
        self.priority_input = QLineEdit()
        
        input_layout.addWidget(QLabel('PID:'))
        input_layout.addWidget(self.pid_input)
        input_layout.addWidget(QLabel('Arrival Time:'))
        input_layout.addWidget(self.arrival_input)
        input_layout.addWidget(QLabel('Burst Time:'))
        input_layout.addWidget(self.burst_input)
        input_layout.addWidget(QLabel('Priority:'))
        input_layout.addWidget(self.priority_input)
        
        add_button = QPushButton('Add Process')
        add_button.clicked.connect(self.add_process)
        input_layout.addWidget(add_button)
        
        layout.addLayout(input_layout)
        
        # Algorithm selection
        algo_layout = QHBoxLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['FCFS', 'SJF', 'Priority', 'Round Robin'])
        algo_layout.addWidget(QLabel('Scheduling Algorithm:'))
        algo_layout.addWidget(self.algo_combo)
        
        self.quantum_input = QLineEdit()
        self.quantum_input.setText('2')
        algo_layout.addWidget(QLabel('Quantum (for RR):'))
        algo_layout.addWidget(self.quantum_input)
        
        run_button = QPushButton('Run Scheduler')
        run_button.clicked.connect(self.run_scheduler)
        algo_layout.addWidget(run_button)
        
        layout.addLayout(algo_layout)
        
        # Process table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(4)
        self.process_table.setHorizontalHeaderLabels(['PID', 'Arrival Time', 'Burst Time', 'Priority'])
        layout.addWidget(self.process_table)
        
        # Visualization area
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Metrics display
        metrics_layout = QHBoxLayout()
        self.avg_waiting_label = QLabel('Average Waiting Time: -')
        self.avg_turnaround_label = QLabel('Average Turnaround Time: -')
        metrics_layout.addWidget(self.avg_waiting_label)
        metrics_layout.addWidget(self.avg_turnaround_label)
        layout.addLayout(metrics_layout)
        
    def add_process(self):
        try:
            pid = int(self.pid_input.text())
            arrival = int(self.arrival_input.text())
            burst = int(self.burst_input.text())
            priority = int(self.priority_input.text())
            
            process = Process(pid, arrival, burst, priority)
            self.scheduler.add_process(process)
            
            # Add to table
            row = self.process_table.rowCount()
            self.process_table.insertRow(row)
            self.process_table.setItem(row, 0, QTableWidgetItem(str(pid)))
            self.process_table.setItem(row, 1, QTableWidgetItem(str(arrival)))
            self.process_table.setItem(row, 2, QTableWidgetItem(str(burst)))
            self.process_table.setItem(row, 3, QTableWidgetItem(str(priority)))
            
            # Clear input fields
            self.pid_input.clear()
            self.arrival_input.clear()
            self.burst_input.clear()
            self.priority_input.clear()
            
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter valid numeric values')
            
    def run_scheduler(self):
        if not self.scheduler.processes:
            QMessageBox.warning(self, 'Error', 'Please add at least one process')
            return
            
        algorithm = self.algo_combo.currentText()
        
        if algorithm == 'Round Robin':
            try:
                quantum = int(self.quantum_input.text())
                timeline = self.scheduler.round_robin(quantum)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Please enter a valid quantum value')
                return
        else:
            if algorithm == 'FCFS':
                timeline = self.scheduler.fcfs()
            elif algorithm == 'SJF':
                timeline = self.scheduler.sjf()
            elif algorithm == 'Priority':
                timeline = self.scheduler.priority()
                
        # Calculate metrics
        avg_waiting, avg_turnaround = self.scheduler.calculate_metrics()
        self.avg_waiting_label.setText(f'Average Waiting Time: {avg_waiting:.2f}')
        self.avg_turnaround_label.setText(f'Average Turnaround Time: {avg_turnaround:.2f}')
        
        # Visualize timeline
        self.ax.clear()
        self.ax.set_title(f'{algorithm} Scheduling Timeline')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Process')
        
        # Create y-axis labels
        processes = sorted(set(pid for pid, _, _ in timeline))
        y_ticks = range(len(processes))
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels([f'P{pid}' for pid in processes])
        
        # Plot timeline
        for pid, start, end in timeline:
            y = processes.index(pid)
            self.ax.broken_barh([(start, end - start)], (y - 0.4, 0.8), facecolors='tab:blue')
            self.ax.text(start + (end - start) / 2, y, f'P{pid}', ha='center', va='center')
            
        self.canvas.draw() 