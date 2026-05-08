import pickle
import os

def save_object(obj, filepath):
    """Saves a Python object (model, scaler, etc.) to a file using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        
    from colors import GREEN, RESET
    print(f"    {GREEN}[Saved] {filepath}{RESET}")

def load_object(filepath):
    """Loads a Python object from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find the saved object at {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
