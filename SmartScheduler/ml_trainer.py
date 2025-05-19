import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from scheduler_utils import Process, extract_features, average_metrics

def generate_training_data(num_samples=1000):
    """Generate synthetic training data for process scheduling"""
    data = []
    
    for _ in range(num_samples):
        # Generate random process set characteristics
        n_proc = np.random.randint(3, 10)  # 3-9 processes
        avg_burst = np.random.randint(5, 20)  # 5-19 time units
        std_burst = np.random.randint(1, 5)  # 1-4 time units
        avg_priority = np.random.randint(1, 5)  # 1-4
        std_priority = np.random.randint(1, 3)  # 1-2
        arrival_skew = np.random.randint(2, 10)  # 2-9 time units
        
        # Generate process set
        processes = []
        for i in range(n_proc):
            bt = max(1, int(np.random.normal(avg_burst, std_burst)))
            pr = max(1, min(5, int(np.random.normal(avg_priority, std_priority))))
            at = int(np.random.uniform(0, arrival_skew))
            processes.append(Process(i + 1, at, bt, pr))
        
        # Extract features
        features = extract_features(processes)
        
        # Determine best algorithm
        best_algo = find_best_algorithm(processes)
        
        # Add to dataset
        data.append({
            'n_proc': features['n_proc'],
            'avg_arrival': features['avg_arrival'],
            'burst_std': features['burst_std'],
            'has_priority': features['has_priority'],
            'burst_variance': features['burst_variance'],
            'avg_burst': features['avg_burst'],
            'min_burst': features['min_burst'],
            'max_burst': features['max_burst'],
            'max_waiting': features['max_waiting'],
            'time_quantum': features['time_quantum'],
            'best_algorithm': best_algo
        })
    
    return pd.DataFrame(data)

def find_best_algorithm(processes):
    """Find the best algorithm for a set of processes"""
    from Smart_scheduler import fcfs, sjf, round_robin, priority_scheduling
    
    # Run all algorithms
    algos = {
        "FCFS": fcfs,
        "SJF": sjf,
        "RR": lambda p: round_robin(p, quantum=2),
        "Priority": priority_scheduling
    }
    
    best_algo = None
    best_metric = float('inf')
    
    for name, algo in algos.items():
        procs = processes.copy()
        scheduled, _ = algo(procs)
        avg_tat, avg_wt = average_metrics(scheduled)
        metric = avg_tat + avg_wt
        
        if metric < best_metric:
            best_metric = metric
            best_algo = name
    
    return best_algo

def train_model():
    """Train the ML model and save it"""
    print("Generating training data...")
    df = generate_training_data(1000)
    
    # Split features and target
    X = df.drop('best_algorithm', axis=1)
    y = df['best_algorithm']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    print("-" * 50)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']:20}: {row['importance']:.3f}")
    
    # Save model
    joblib.dump(model, "scheduler_model.pkl")
    print("\nModel saved as scheduler_model.pkl")
    
    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "feature_names.pkl")
    print("Feature names saved as feature_names.pkl")
    
    return model, feature_names

if __name__ == "__main__":
    train_model() 