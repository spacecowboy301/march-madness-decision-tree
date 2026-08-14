from .analyze_factor_importance import main as run_factor_analysis
from .analyze_kenpom_discrepancies import KENPOM_PATH, main as run_kenpom_discrepancy_analysis
from .power_matchup_stack import main as run_power_matchup_analysis


def main() -> None:
    run_factor_analysis()
    run_power_matchup_analysis()
    if KENPOM_PATH.exists():
        run_kenpom_discrepancy_analysis()
    else:
        print("Skipping optional KenPom discrepancy report: no local KenPom cache.")


if __name__ == "__main__":
    main()
