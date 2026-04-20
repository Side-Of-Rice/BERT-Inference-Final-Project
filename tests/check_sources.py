def test_model_training():
    # Quick test of "train.py"
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    current_script_path = Path(__file__).resolve()
    parent_dir = current_script_path.parent
    target_script_path = parent_dir / src / 'data_ingestion.py'
    
    result = subprocess.run([sys.executable, str(target_script_path)], capture_output = True)
    assert result.returncode == 0, "Training script failed!"
