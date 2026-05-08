import os
import time
import sys

# Force working directory to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Src"))

# Enable ANSI colors for Windows CMD
if os.name == 'nt':
    os.system("")

from colors import PRIMARY, INFO, WARNING, DANGER, TEXT, RESET, BOLD

def setup_directories():
    os.makedirs("Plots/regression", exist_ok=True)
    os.makedirs("Plots/classification", exist_ok=True)
    os.makedirs("Tables/regression", exist_ok=True)
    os.makedirs("Tables/classification", exist_ok=True)
    os.makedirs("Models/regression", exist_ok=True)
    os.makedirs("Models/classification", exist_ok=True)

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{PRIMARY}{BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║   [ SYS_INIT ] ONLINE GAMES POPULARITY PREDICTOR           ║")
    print("║   [ STATUS   ] ONLINE                                      ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")

def spinner(text, duration=1.0):
    chars = ["-", "\\", "|", "/"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{INFO}[{chars[i % len(chars)]}] {text}{RESET}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write(f"\r{PRIMARY}[+] {text} COMPLETE            {RESET}\n\n")

def main_menu():
    setup_directories()
    
    try:
        from train import run_pipeline
        from test import test_pipeline
    except ImportError as e:
        print(f"{DANGER}[!] Import Error: {e}{RESET}")
        return

    while True:
        print_header()
        print(f"{TEXT}Select an execution protocol:{RESET}\n")
        print(f"  [1] {BOLD}Train Regression Model{RESET}")
        print(f"  [2] {BOLD}Train Classification Model{RESET}")
        print(f"  [3] {BOLD}Train Both Pipelines{RESET}")
        print(f"  [4] {BOLD}Test Regression Pipeline{RESET}")
        print(f"  [5] {BOLD}Test Classification Pipeline{RESET}")
        print(f"  [6] {BOLD}Exit{RESET}\n")
        
        choice = input(f"{PRIMARY}root:~${RESET} ").strip()
        
        if choice == '1':
            spinner("INITIALIZING REGRESSION PIPELINE")
            run_pipeline(task="regression", train_file="Datasets/regression_train_data.csv", target="RecommendationCount")
            input(f"\n{INFO}Press Enter to return to main menu...{RESET}")
            
        elif choice == '2':
            spinner("INITIALIZING CLASSIFICATION PIPELINE")
            run_pipeline(task="classification", train_file="Datasets/classification_train_data.csv", target="GamePopularity")
            input(f"\n{INFO}Press Enter to return to main menu...{RESET}")
            
        elif choice == '3':
            spinner("INITIALIZING BATCH TRAINING")
            run_pipeline(task="regression", train_file="Datasets/regression_train_data.csv", target="RecommendationCount")
            print(f"\n{WARNING}[*] REGRESSION COMPLETE. STAND BY FOR CLASSIFICATION...{RESET}\n")
            time.sleep(1)
            run_pipeline(task="classification", train_file="Datasets/classification_train_data.csv", target="GamePopularity")
            input(f"\n{INFO}Press Enter to return to main menu...{RESET}")
            
        elif choice == '4':
            spinner("PREPARING REGRESSION TEST ENV")
            test_file = input(f"{WARNING}[?] Enter absolute path to regression testfile.csv:{RESET} ").strip()
            if test_file:
                test_pipeline(test_file=test_file, task="regression")
            input(f"\n{INFO}Press Enter to return to main menu...{RESET}")
            
        elif choice == '5':
            spinner("PREPARING CLASSIFICATION TEST ENV")
            test_file = input(f"{WARNING}[?] Enter absolute path to classification testfile.csv:{RESET} ").strip()
            if test_file:
                test_pipeline(test_file=test_file, task="classification")
            input(f"\n{INFO}Press Enter to return to main menu...{RESET}")
            
        elif choice == '6':
            print(f"\n{DANGER}[!] TERMINATING CONNECTION. GOODBYE.{RESET}")
            break
            
        else:
            print(f"\n{DANGER}[!] INVALID INPUT. PLEASE TRY AGAIN.{RESET}")
            time.sleep(1)

if __name__ == "__main__":
    main_menu()
