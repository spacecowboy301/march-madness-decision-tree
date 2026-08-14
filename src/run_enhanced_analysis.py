from .analyze_factor_importance import main as run_factor_analysis
from .power_matchup_stack import main as run_power_matchup_analysis


def main() -> None:
    run_factor_analysis()
    run_power_matchup_analysis()


if __name__ == "__main__":
    main()
