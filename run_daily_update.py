import subprocess
import sys
from datetime import datetime

def run_step(description, command):
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f" Error during: {description}. Aborting pipeline.")
        sys.exit(result.returncode)

def main():
    start_time = datetime.now()
    today_str = start_time.strftime("%Y-%m-%d")

    # Step 1: Harvest settlement prices from IB Gateway
    run_step("Harvesting Daily Futures Settlements (IBKR)", "python fetch_rates.py")

    # Step 2: Compute curves and meeting probabilities
    run_step("Computing Implied Policy Curves (WLS Engine)", "python rate_engine.py")

    # Step 3: Render Quarto site and push to GitHub Pages
    run_step("Publishing Updated Site to snoquant.com", "quarto publish gh-pages --no-prompt")

    # Step 4: Back up source code to GitHub main branch
    git_commands = (
        f'git add . && '
        f'git commit -m "Daily automated rates update: {today_str}" && '
        f'git push origin main'
    )
    run_step("Backing Up Source Code to Git", git_commands)

    elapsed = datetime.now() - start_time
    print(f"\n🎉 Pipeline completed successfully in {elapsed.total_seconds():.1f} seconds!")

if __name__ == "__main__":
    main()