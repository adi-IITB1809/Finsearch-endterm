import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["01_data_collection.py"],
    ["02_preprocessing.py"],
    ["03_split_data.py"],
    ["rl_agent.py", "--steps", "400000"],
    ["lstm_agent.py"],
    ["arima_agent.py"],
    ["eval_compare.py"],
    ["eval_compare.py", "--recent"],
]

for step in STEPS:
    print("\n" + "=" * 70 + f"\n>>> python {' '.join(step)}\n" + "=" * 70, flush=True)
    r = subprocess.run([sys.executable] + step, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"Step failed: {step}")
print("\nAll steps done. See results/")
