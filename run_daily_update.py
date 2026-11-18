import subprocess
import sys
from datetime import datetime

def run_step(description, command_list):
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"❌ Error during: {description}. Aborting pipeline.")
        sys.exit(result.returncode)

def main():
    start_time = datetime.now()
    today_str = start_time.strftime("%Y-%m-%d")
    py = sys.executable  # Automatically uses your active python3 environment

    # Step 1: Harvest settlement prices from IB Gateway
    run_step("Harvesting Daily Futures Settlements (IBKR)", [py, "fetch_rates.py"])

    # Step 2: Run all 7 WLS Central Bank Rate Engines
    engines = [
        ("Federal Reserve (Fed)", "fed_rate_engine.py"),
        ("European Central Bank (ECB)", "ecb_rate_engine.py"),
        ("Bank of England (BoE)", "boe_rate_engine.py"),
        ("Bank of Japan (BoJ)", "boj_rate_engine.py"),
        ("Bank of Canada (BoC)", "boc_rate_engine.py"),
        ("Reserve Bank of Australia (RBA)", "rba_rate_engine.py"),
        ("Swiss National Bank (SNB)", "snb_rate_engine.py"),
    ]
    for bank_name, script in engines:
        run_step(f"Computing Implied Policy Curve: {bank_name}", [py, script])

    # Step 3: Render Quarto site and push to GitHub Pages
    run_step(
        "Publishing Updated Site to snoquant.com",
        ["quarto", "publish", "gh-pages", "--no-prompt", "--no-browser"]
    )

    # Step 4: Back up updated CSVs to GitHub main branch
    run_step("Staging Updated Data CSVs", ["git", "add", "data/*.csv"])
    subprocess.run(["git", "commit", "-m", f"Daily automated rates update: {today_str}"], check=False)
    run_step("Pushing Data Backup to GitHub main", ["git", "push", "origin", "main"])

    elapsed = datetime.now() - start_time
    print(f"\n🎉 Pipeline completed successfully in {elapsed.total_seconds():.1f} seconds!")

if __name__ == "__main__":
    main()