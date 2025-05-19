# Smart Scheduler

A machine learning-based process scheduling simulator that helps determine the most efficient scheduling algorithm for a given set of processes.

## Features

- Interactive GUI for process management
- Support for multiple scheduling algorithms:
  - First Come First Serve (FCFS)
  - Shortest Job First (SJF)
  - Round Robin (RR)
  - Priority Scheduling
- Machine Learning-based algorithm prediction
- Visual Gantt charts for process scheduling
- Dataset testing capabilities
- Real-time performance metrics

## Requirements

- Python 3.x
- Required Python packages:
  - tkinter
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - joblib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SmartScheduler
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python scheduler_gui.py
```

2. Using the GUI:
   - Add processes manually by entering:
     - Arrival Time
     - Burst Time
     - Priority
   - Or load processes from the dataset using "Load from Dataset"
   - Click "Predict Best Algorithm" to get ML-based recommendations
   - View Gantt charts and performance metrics for all algorithms

## Project Structure

- `scheduler_gui.py`: Main GUI application
- `Smart_scheduler.py`: Core scheduling algorithms implementation
- `ml_trainer.py`: Machine learning model training
- `ml_predictor.py`: ML model for algorithm prediction
- `scheduler_utils.py`: Utility functions and helper classes
- `scheduler_dataset.csv`: Training dataset
- `scheduler_model.pkl`: Trained ML model
- `feature_names.pkl`: Feature names for the ML model

## How It Works

1. **Process Input**: Users can input processes manually or load from the dataset
2. **Feature Extraction**: System extracts relevant features from the process set
3. **ML Prediction**: Machine learning model predicts the best scheduling algorithm
4. **Algorithm Execution**: All algorithms are run on the process set
5. **Performance Analysis**: Results are displayed with Gantt charts and metrics

## Performance Metrics

The system evaluates algorithms based on:
- Average Turnaround Time
- Average Waiting Time
- Total Metric (Turnaround Time + Waiting Time)

## Contributing

Feel free to submit issues and enhancement requests! 