import joblib
import numpy as np
from scheduler_utils import Process, extract_features

class SchedulerPredictor:
    def __init__(self):
        try:
            self.model = joblib.load("scheduler_model.pkl")
            self.feature_names = joblib.load("feature_names.pkl")
        except FileNotFoundError:
            print("Model files not found. Please run ml_trainer.py first.")
            raise

    def predict_best_algorithm(self, processes):
        """Predict the best scheduling algorithm for a set of processes"""
        # Extract features
        features = extract_features(processes)
        
        # Convert features to numpy array in correct order
        X = np.array([[features[name] for name in self.feature_names]])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get confidence scores for each algorithm
        algo_probs = dict(zip(self.model.classes_, probabilities))
        
        return {
            'best_algorithm': prediction,
            'confidence_scores': algo_probs
        }

def main():
    # Example usage
    predictor = SchedulerPredictor()
    
    # Example process set
    processes = [
        Process(1, 0, 4, 2),
        Process(2, 1, 3, 1),
        Process(3, 2, 5, 3),
        Process(4, 3, 2, 2)
    ]
    
    # Get prediction
    result = predictor.predict_best_algorithm(processes)
    
    print("\nML Prediction Results:")
    print("=" * 50)
    print(f"Best Algorithm: {result['best_algorithm']}")
    print("\nConfidence Scores:")
    for algo, score in result['confidence_scores'].items():
        print(f"{algo:10}: {score:.2%}")

if __name__ == "__main__":
    main() 