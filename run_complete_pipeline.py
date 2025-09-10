#!/usr/bin/env python3
"""
Complete Football Prediction AI Pipeline
Inspired by 85% accuracy tennis predictions

This script runs the entire pipeline:
1. Data collection
2. Feature engineering  
3. Model training
4. World Cup predictions
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        # Change to src directory to run scripts
        original_dir = os.getcwd()
        src_dir = os.path.join(os.path.dirname(__file__), 'src')
        os.chdir(src_dir)
        
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        print("✅ SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ ERROR!")
        print("Error output:", e.stderr)
        if e.stdout:
            print("Standard output:", e.stdout)
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False
        
    finally:
        os.chdir(original_dir)

def main():
    print("🏆 FOOTBALL PREDICTION AI - COMPLETE PIPELINE 🏆")
    print("Targeting 85%+ accuracy like tennis predictions")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    pipeline_steps = [
        ('data_collector.py', 'Collecting Football Data'),
        ('feature_engineering.py', 'Engineering Features'),
        ('train_models.py', 'Training AI Models'),
        ('world_cup_predictor.py', 'Predicting World Cup')
    ]
    
    success_count = 0
    total_steps = len(pipeline_steps)
    
    for script, description in pipeline_steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"❌ Pipeline failed at step: {description}")
            break
    
    print(f"\n{'='*60}")
    print("🏁 PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/{total_steps} steps")
    print(f"Success Rate: {success_count/total_steps*100:.1f}%")
    
    if success_count == total_steps:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Check the results/ folder for World Cup predictions")
        print("2. Open the Jupyter notebook for detailed analysis")
        print("3. Run individual prediction scripts as needed")
        print("\nUsage examples:")
        print("• python src/predict.py - Interactive prediction tool")
        print("• python src/world_cup_predictor.py - Full tournament prediction")
        print("• jupyter notebook notebooks/analysis_and_visualization.ipynb - Detailed analysis")
    else:
        print("❌ Pipeline incomplete. Please check the errors above.")
        print("You may need to:")
        print("• Install missing dependencies: pip install -r requirements.txt")
        print("• Check data sources and API keys")
        print("• Ensure proper file permissions")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()