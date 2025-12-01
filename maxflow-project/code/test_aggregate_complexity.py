import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(PROJECT_ROOT, 'results')
SUMMARY = os.path.join(RESULTS, 'empirical_complexity_summary.csv')

REQUIRED_COLS = {
    'Algorithm',
    'slope_time_vs_n_sparse',
    'slope_time_vs_n_dense',
    'slope_time_vs_m_n200',
    'slope_time_vs_density_n200',
    'slope_time_vs_Cmax'
}

def test_summary_exists_and_columns():
    assert os.path.exists(SUMMARY), f"Missing summary CSV: {SUMMARY}"
    df = pd.read_csv(SUMMARY)
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"
    assert df['Algorithm'].nunique() >= 5, "Expected at least 5 algorithms in summary"

if __name__ == '__main__':
    # simple manual run
    test_summary_exists_and_columns()
    print('aggregate complexity summary test passed')
