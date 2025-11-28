"""
Validation script to check project setup and dependencies.
"""
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'yaml': 'pyyaml'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True

def check_project_structure():
    """Check if all required directories and files exist."""
    print("\nChecking project structure...")
    
    required_paths = [
        "src/data/data_loader.py",
        "src/data/preprocessor.py",
        "src/models/neural_network.py",
        "src/models/model_builder.py",
        "src/training/trainer.py",
        "src/training/cross_validator.py",
        "src/evaluation/metrics.py",
        "src/utils/config.py",
        "src/utils/seed.py",
        "config/config.yaml",
        "main.py",
        "requirements.txt"
    ]
    
    missing = []
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✓ {path_str}")
        else:
            print(f"✗ {path_str} is missing")
            missing.append(path_str)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All required files exist!")
        return True

def check_directories():
    """Check if required output directories exist."""
    print("\nChecking output directories...")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "models/saved_models",
        "results/logs",
        "results/plots"
    ]
    
    for dir_str in required_dirs:
        dir_path = Path(dir_str)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ {dir_str}")
        elif dir_path.exists() and not dir_path.is_dir():
            print(f"⚠ {dir_str} exists but is not a directory")
        else:
            print(f"Creating {dir_str}...")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created {dir_str}")
            except Exception as e:
                print(f"✗ Failed to create {dir_str}: {e}")
                return False
    
    print("\n✓ All directories ready!")
    return True

def check_config():
    """Check if configuration file is valid."""
    print("\nChecking configuration...")
    
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['seed', 'data', 'model', 'training', 'cross_validation', 'paths']
        for key in required_keys:
            if key in config:
                print(f"✓ Config has '{key}' section")
            else:
                print(f"✗ Config missing '{key}' section")
                return False
        
        print("\n✓ Configuration is valid!")
        return True
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False

def main():
    """Run all validation checks."""
    print("="*60)
    print("Project Validation")
    print("="*60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Output Directories", check_directories),
        ("Configuration", check_config)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        print("You can run 'python main.py' to train the model.")
        return 0
    else:
        print("\n✗ Some validation checks failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
