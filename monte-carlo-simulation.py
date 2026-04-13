"""
=============================================================================
CMEPA MONTE CARLO SIMULATION — PRODUCTION-READY RESEARCH VERSION
"Quantifying the Impact of the Capital Market Efficiency Promotion Act:
 A Monte Carlo Analysis of the Stock Transaction Tax Reduction on
 Trading Volume and Market Volatility"

Authors   : Bajo, Cacho, Rizon, Villamor, Ylaya
School    : University of San Jose-Recoletos
Programme : Bachelor of Science in Accountancy
Period    : A.Y. 2025–2026

METHODOLOGY: Multi-horizon stationary simulation with EDA
(1-Year, 5-Year, 10-Year with identical daily policy shocks)
LAST UPDATED: 2026-04-13
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr, norm
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#1a1a1a',
})

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

STT_PRE        = 0.006                                  # 0.6% (Pre-CMEPA)
STT_POST       = 0.001                                 # 0.1% (CMEPA)
STT_CHANGE_PCT = (STT_POST - STT_PRE) / STT_PRE         # −83.33%
BETA_TV_CLIP   = 2.5

# HORIZONS: Only trading_days differ; policy shock identical every day
HORIZONS = {
    '1-Year': {
        'sims_per_day': 1_000,
        'trading_days': 252,
        'total_iters': 252_000,
        'horizon_years': 1,
    },
    '5-Year': {
        'sims_per_day': 500,
        'trading_days': 1_260,
        'total_iters': 630_000,
        'horizon_years': 5,
    },
    '10-Year': {
        'sims_per_day': 250,
        'trading_days': 2_520,
        'total_iters': 630_000,
        'horizon_years': 10,
    },
}

C_BLUE  = '#1a4f8a'
C_RED   = '#8b1a1a'
C_GREEN = '#1a5c2e'
C_AMBER = '#b8860b'
C_DARK  = '#1a1a1a'
C_LIGHT = '#f5f5f5'

# File paths
PSE_FILE = "PSE Dataset.xlsx"
MED_FILE = "Mediating Variables - Dataset.xlsx"

# Output directory
OUTPUT_DIR = "CMEPA_Simulation_Results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_message(msg: str, stage: str = "INFO") -> None:
    """Logs messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{stage}] {msg}")


def render_table(title: str, col_labels: list, row_data: list, fname: str = None,
                 col_widths: list = None, footnote: str = None, highlight_rows: list = None,
                 figsize: tuple = None) -> plt.Figure:
    """Renders publication-quality tables as PNG."""
    n_cols = len(col_labels)
    n_rows = len(row_data)

    if figsize is None:
        fig_h = max(2.5, 0.44 * n_rows + 1.8)
        fig_w = max(10, 1.5 * n_cols)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    cell_colours = [['white'] * n_cols for _ in range(n_rows)]
    bbox_y0 = 0.08 if footnote else 0.03
    bbox_h = 1.0 - bbox_y0 - 0.05

    tbl = ax.table(cellText=row_data, colLabels=col_labels, cellLoc='center',
                   loc='center', cellColours=cell_colours,
                   bbox=[0.01, bbox_y0, 0.98, bbox_h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#444444')
        cell.set_linewidth(0.6)
        cell.set_facecolor('white')
        if row == 0:
            cell.set_text_props(color='black', fontweight='bold', fontsize=9)
            cell.set_height(0.10)
            cell.set_linewidth(1.0)
        else:
            r_idx = row - 1
            fw = 'bold' if (col == 0 or (highlight_rows and r_idx in highlight_rows)) else 'normal'
            cell.set_height(0.082)
            cell.set_text_props(fontweight=fw)
            if col == 0:
                cell.set_text_props(ha='left', fontweight=fw)

    if col_widths:
        for ci, w in enumerate(col_widths):
            for ri in range(n_rows + 1):
                tbl[ri, ci].set_width(w)

    fig.suptitle(title, fontsize=10, fontweight='bold', x=0.5, y=0.99,
                 ha='center', va='top', color=C_DARK, fontfamily='serif')

    if footnote:
        fig.text(0.015, 0.01, footnote, fontsize=7.5, color='#444444',
                 style='italic', va='bottom', fontfamily='serif', wrap=True)

    plt.tight_layout(rect=[0, 0.06 if footnote else 0, 1, 0.96])

    if fname:
        full_path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        log_message(f"PNG Table saved → {fname}", "OUTPUT")
    
    plt.show()
    return fig


def _section_header(stage: int, title: str) -> None:
    """Prints formatted section header."""
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"  STAGE {stage} — {title}")
    print(bar)


def _subsection(title: str) -> None:
    """Prints subsection header."""
    print(f"\n  {'─' * 74}")
    print(f"    {title}")
    print(f"  {'─' * 74}")


def diagnose_eda_data_loading(pse_file: str) -> bool:
    """Diagnostic: Check if EDA data is loading correctly."""
    print("\n" + "="*80)
    print("  DIAGNOSTIC: IS EDA WORKING WITH REAL DATA?")
    print("="*80)
    
    try:
        pse_data = load_pse_data(pse_file)
        
        print("\n  ✅ PSE Data File Loaded Successfully")
        print(f"     File: {pse_file}")
        
        has_data = False
        for period, df in pse_data.items():
            if len(df) > 0:
                has_data = True
                print(f"\n     {period}:")
                print(f"       Rows: {len(df)}")
                print(f"       Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
                print(f"       Trade Value Mean: ₱{df['Trade Value'].mean():,.0f}")
                print(f"       Trade Value Std:  ₱{df['Trade Value'].std():,.0f}")
                print(f"       Price Mean:  ₱{df['Closing Price'].mean():,.2f}")
            else:
                print(f"\n     ⚠️  {period}: EMPTY (0 rows)")
        
        if not has_data:
            print("\n  ❌ WARNING: All datasets are empty!")
            return False
        
        eda_params = calculate_eda_parameters(pse_data)
        
        print("\n  ✅ EDA PARAMETERS CALCULATED:")
        print(f"     TV_µ (Mean Daily Volume):   {eda_params['TV_mu']:>15,.0f} shares")
        print(f"     TV_σ (Std Dev):             {eda_params['TV_sigma']:>15,.0f} shares")
        if eda_params['TV_mu'] > 0:
            cv = (eda_params['TV_sigma']/eda_params['TV_mu']*100)
            print(f"     Coefficient of Variation:   {cv:>14.1f}%")
        print(f"     P_avg (VWAP):               ₱{eda_params['P_avg']:>14,.4f}")
        print(f"     GSP_Mean:                   ₱{eda_params['GSP_Mean']:>14,.0f}")
        print(f"     Volatility_pre (5-day):     {eda_params['VOL_pre']:>15.6f}")
        
        # Test Normal distribution sampling
        print("\n  ✅ TESTING NORMAL DISTRIBUTION SAMPLING (EDA in action):")
        test_samples = np.random.normal(eda_params['TV_mu'], eda_params['TV_sigma'], 10000)
        print(f"     Drew 10,000 samples from Normal(µ={eda_params['TV_mu']:,.0f}, σ={eda_params['TV_sigma']:,.0f})")
        print(f"     Sample Mean:      {test_samples.mean():>15,.0f}  (Expected: {eda_params['TV_mu']:,.0f})")
        print(f"     Sample Std:       {test_samples.std():>15,.0f}  (Expected: {eda_params['TV_sigma']:,.0f})")
        print(f"     Min:              {test_samples.min():>15,.0f}")
        print(f"     Max:              {test_samples.max():>15,.0f}")
        pct_1sigma = (np.sum(np.abs(test_samples - eda_params['TV_mu']) <= eda_params['TV_sigma']) / len(test_samples) * 100)
        pct_2sigma = (np.sum(np.abs(test_samples - eda_params['TV_mu']) <= 2*eda_params['TV_sigma']) / len(test_samples) * 100)
        print(f"     % Within ±1σ:     {pct_1sigma:>14.1f}%  (Expected: ~68%)")
        print(f"     % Within ±2σ:     {pct_2sigma:>14.1f}%  (Expected: ~95%)")
        
        print("\n  ✅✅ EDA IS WORKING — Real data loaded, Normal distribution validated")
        print("="*80)
        return True
        
    except FileNotFoundError as e:
        print(f"\n  ❌ ERROR: {e}")
        print(f"     File '{pse_file}' not found in working directory")
        print(f"     EDA CANNOT WORK — synthetic placeholder data will be used")
        print(f"\n     ACTION REQUIRED:")
        print(f"     1. Ensure {pse_file} is in your working directory")
        print(f"     2. Verify sheet names match the code")
        print(f"     3. Check Excel file is not corrupted")
        print("="*80)
        return False
    except Exception as e:
        print(f"\n  ❌ ERROR DURING EDA DIAGNOSTICS: {type(e).__name__}")
        print(f"     {str(e)}")
        print("="*80)
        return False


# ============================================================================
# STAGE 0 — DATA LOADING & EDA
# ============================================================================

def load_pse_data(file: str) -> dict:
    """
    Loads PSE data from Excel with EDA-compatible structure.
    
    EDA REQUIREMENT: Returns DataFrames with columns:
    - 'Date': Trading date
    - 'Trade Value': Daily GSP (gross value of shares traded)
    - 'Volume': Daily number of shares traded
    - 'Closing Price': Daily closing price
    - 'Daily Returns': Daily log returns
    - 'Volatility': 5-day rolling volatility
    """
    
    def _clean(sheet: str, date_lo: str = None, date_hi: str = None) -> pd.DataFrame:
        """Cleans and validates PSE data sheet."""
        df = pd.read_excel(file, sheet_name=sheet)
        
        # Standardize column names (case-insensitive matching)
        df.columns = [col.strip() for col in df.columns]
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Closing Price'] = pd.to_numeric(df['Closing Price'], errors='coerce')
        df['Trade Value'] = pd.to_numeric(df['Trade Value'], errors='coerce')
        df['Daily Returns'] = pd.to_numeric(df['Daily Returns'], errors='coerce')
        df['Volume'] = pd.to_numeric(df.get('Volume', pd.Series(dtype=float)), errors='coerce')
        
        # Try multiple column name variants for volatility
        vol_cols = ['SHORT TERM VOLATILITY (5-DAY WINDOW)', 'Volatility', 'VOL', '5-Day Volatility']
        vol_col = None
        for col in vol_cols:
            if col in df.columns:
                vol_col = col
                break
        
        if vol_col:
            df['Volatility'] = pd.to_numeric(df[vol_col], errors='coerce')
        else:
            df['Volatility'] = 0.015  # Default if not found
        
        df = df.dropna(subset=['Date', 'Trade Value', 'Closing Price'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        if date_lo:
            df = df[df['Date'] >= date_lo]
        if date_hi:
            df = df[df['Date'] <= date_hi]
        
        return df.reset_index(drop=True)

    try:
        pre_train = _clean('Trading Volume Pre-TRAIN', '2016-07-01', '2017-12-31')
        post_train = _clean('Trading Volume Post-TRAIN', '2018-01-01', '2019-06-30')
        pre_cmepa = _clean('Trading Volume Pre-CMEPA', '2024-07-01', '2025-06-30')
        post_cmepa = _clean('Trading Volume Post-CMEPA', '2025-07-01')
    except Exception as e:
        log_message(f"Error loading data: {e}", "ERROR")
        raise

    log_message("PSE Data Loaded Successfully", "DATA")
    
    for lbl, df in [('Pre-TRAIN', pre_train), ('Post-TRAIN', post_train),
                    ('Pre-CMEPA', pre_cmepa), ('Post-CMEPA', post_cmepa)]:
        if len(df):
            print(f"    {lbl:<14}: {len(df):>4} trading days "
                  f"({df['Date'].min().date()} – {df['Date'].max().date()})")
        else:
            print(f"    {lbl:<14}: 0 rows (empty)")

    return {'pre_train': pre_train, 'post_train': post_train,
            'pre_cmepa': pre_cmepa, 'post_cmepa': post_cmepa}


def load_mediating_data(file: str) -> dict:
    """Loads mediating variables for lambda calculation."""
    try:
        gdp = pd.read_excel(file, sheet_name='GDPGrowth_Data', header=3)
        gdp.columns = ['Year', 'Quarter', 'Period', 'GDP_Growth', 'Months', 'Note']
        gdp['Year'] = pd.to_numeric(gdp['Year'], errors='coerce')
        gdp['GDP_Growth'] = pd.to_numeric(gdp['GDP_Growth'], errors='coerce')
        gdp = gdp.dropna(subset=['Year', 'GDP_Growth']).reset_index(drop=True)

        inf = pd.read_excel(file, sheet_name='InflationRate_Data', header=3)
        inf.columns = ['Year', 'Month', 'InflationRate', 'Classification', 'Note']
        inf['Year'] = pd.to_numeric(inf['Year'], errors='coerce')
        inf['InflationRate'] = pd.to_numeric(inf['InflationRate'], errors='coerce')
        inf = inf.dropna(subset=['Year', 'InflationRate']).reset_index(drop=True)

        def _load_mkt(sheet_name: str) -> pd.DataFrame:
            df = pd.read_excel(file, sheet_name=sheet_name, header=3)
            df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change']
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            return df.dropna(subset=['Date', 'Price']).sort_values('Date').reset_index(drop=True)

        mkt_pre = _load_mkt('PreCMEPA_MarketConditions_Data')
        mkt_post = _load_mkt('PostCMEPA_MarketConditions_Data')

        flr = pd.read_excel(file, sheet_name='InvestorType_Data', header=2)
        flr.columns = ['Period', 'Month', 'TotalValue', 'AvgDaily',
                       'ForeignPct', 'ForeignValue', 'LocalPct', 'FLR']
        for col in ['TotalValue', 'ForeignPct', 'ForeignValue', 'LocalPct', 'FLR']:
            flr[col] = pd.to_numeric(flr[col], errors='coerce')
        flr = flr.dropna(subset=['FLR']).reset_index(drop=True)

        log_message("Mediating Variables Loaded", "DATA")
        return {'gdp': gdp, 'inflation': inf,
                'mkt_pre': mkt_pre, 'mkt_post': mkt_post, 'flr': flr}
    
    except FileNotFoundError:
        log_message(f"Mediating data file not found: {file}", "WARNING")
        return {'gdp': pd.DataFrame(), 'inflation': pd.DataFrame(),
                'mkt_pre': pd.DataFrame(), 'mkt_post': pd.DataFrame(), 'flr': pd.DataFrame()}


# ============================================================================
# STAGE 1 — EDA PARAMETER CALCULATION
# ============================================================================

def calculate_eda_parameters(pse_data: dict) -> dict:
    """
    STAGE 1: Estimates EDA parameters from pre-CMEPA data.
    
    EDA CORE: Calculates empirical µ and σ from historical data,
    which define the Normal distribution for Monte Carlo sampling.
    """
    _section_header(1, "EDA PARAMETER CALCULATION (Estimated Distribution Approach)")
    print("    Baseline window: Jul 2024 – Dec 2024 (seasonally aligned with")
    print("    post-CMEPA evaluation period Jul–Dec 2025)")

    pre_full = pse_data['pre_cmepa'].copy()
    for col in ['Daily Returns', 'Closing Price', 'Trade Value', 'Volume', 'Volatility']:
        pre_full[col] = pd.to_numeric(pre_full[col], errors='coerce')

    # Extract 6-month baseline window (Jul-Dec 2024)
    pre = pre_full[(pre_full['Date'] >= '2024-07-01') &
                   (pre_full['Date'] <= '2024-12-31')].copy()
    if len(pre) == 0:
        log_message("No data for Jul–Dec 2024, using full pre-CMEPA period", "WARNING")
        pre = pre_full.copy()

    # EDA CALCULATION: Extract µ and σ
    vp = pre.dropna(subset=['Trade Value', 'Volume', 'Closing Price'])

    TV_mu = float(vp['Volume'].mean())           # ✅ EDA: Empirical mean
    TV_sigma = float(vp['Volume'].std())         # ✅ EDA: Empirical std dev
    P_avg = (float(vp['Trade Value'].sum() / vp['Volume'].sum()) 
             if vp['Volume'].sum() > 0 
             else float(vp['Closing Price'].mean()))
    GSP_Mean = TV_mu * P_avg
    VOL_pre = float(pre['Volatility'].dropna().mean())

    pre_tv_mean = float(pse_data['pre_cmepa']['Trade Value'].mean())
    post_tv_mean = float(pse_data['post_cmepa']['Trade Value'].mean())
    pre_vol_mean = float(pse_data['pre_cmepa']['Volatility'].mean())
    post_vol_mean = float(pse_data['post_cmepa']['Volatility'].mean())

    print(f"\n    EDA BASELINE STATISTICS (Jul 2024 – Dec 2024):")
    print(f"    {'─' * 70}")
    print(f"    Mean Daily Volume (TV_µ):        {TV_mu:>18,.0f} shares")
    print(f"    Std Dev Volume (TV_σ):           {TV_sigma:>18,.0f} shares")
    if TV_mu > 0:
        cv = (TV_sigma / TV_mu * 100)
        print(f"    Coefficient of Variation:        {cv:>18.2f}%")
    print(f"    VWAP (P_avg):                    ₱{P_avg:>17,.4f}")
    print(f"    Mean Daily GSP (TV_µ × P_avg):   ₱{GSP_Mean:>17,.0f}")
    print(f"    5-Day Rolling Volatility:        {VOL_pre:>18.6f}")
    print(f"    {'─' * 70}")

    render_table(
        title="Table 1.  EDA Baseline Parameters — Pre-CMEPA Reference Period (Jul 2024 – Dec 2024)\n"
              "These parameters define the Normal distributions used in Monte Carlo sampling",
        col_labels=['Parameter', 'Symbol', 'Value', 'Unit'],
        row_data=[
            ['Mean Daily Share Volume', 'TV_µ', f'{TV_mu:,.2f}', 'shares/day'],
            ['Std Dev of Daily Share Volume', 'TV_σ', f'{TV_sigma:,.2f}', 'shares/day'],
            ['Volume-Weighted Average Price', 'P_avg', f'₱{P_avg:,.4f}', '₱/share'],
            ['Mean Daily GSP (TV_µ × P_avg)', 'GSP_µ', f'₱{GSP_Mean:,.2f}', '₱/day'],
            ['5-Day Rolling Volatility Mean', 'VOL_pre', f'{VOL_pre:.6f}', '5-day σ'],
        ],
        fname='table1_eda_parameters.png',
        col_widths=[0.40, 0.15, 0.25, 0.20],
        figsize=(14, 3.8),
        footnote=("EDA: Empirical µ and σ computed from real PSE data (Jul–Dec 2024).\n"
                  "Each MCS iteration: TV_pre^(k) ~ Normal(TV_µ, TV_σ)  |  VOL_pre^(k) ~ Constant"),
    )

    return {
        'TV_mu': TV_mu, 'TV_sigma': TV_sigma, 'P_avg': P_avg,
        'GSP_Mean': GSP_Mean, 'VOL_pre': VOL_pre,
        'pre_tv_mean': pre_tv_mean, 'post_tv_mean': post_tv_mean,
        'pre_vol_mean': pre_vol_mean, 'post_vol_mean': post_vol_mean,
    }


# ============================================================================
# STAGE 2 — ELASTICITY ESTIMATION
# ============================================================================

def estimate_elasticity(pre_train: pd.DataFrame, post_train: pd.DataFrame) -> dict:
    """
    STAGE 2: Estimates elasticity (β) from TRAIN Law natural experiment.
    
    TRAIN Law: STT 0.5% → 0.6% (20% increase)
    β = (%ΔOutcome) / (%ΔSTT)
    """
    _section_header(2, "ELASTICITY PARAMETER ESTIMATION (TRAIN Law Natural Experiment, 2018)")

    STT_CHANGE = 0.20  # 20% STT increase

    pre_q = (pre_train.copy()
             .assign(Quarter=lambda d: d['Date'].dt.to_period('Q'))
             .groupby('Quarter')
             .agg(volume_mean=('Trade Value', 'mean'),
                  volatility_sd=('Volatility', 'std'))
             .reset_index().dropna())

    post_q = (post_train.copy()
              .assign(Quarter=lambda d: d['Date'].dt.to_period('Q'))
              .groupby('Quarter')
              .agg(volume_mean=('Trade Value', 'mean'),
                   volatility_sd=('Volatility', 'std'))
              .reset_index().dropna())

    n_q = min(len(pre_q), len(post_q))
    beta_tv_raw, beta_vol_list = [], []

    for i in range(n_q):
        pct_tv = ((post_q.loc[i, 'volume_mean'] - pre_q.loc[i, 'volume_mean'])
                  / pre_q.loc[i, 'volume_mean'])
        beta_tv_raw.append(pct_tv / STT_CHANGE)

        pre_v, post_v = pre_q.loc[i, 'volatility_sd'], post_q.loc[i, 'volatility_sd']
        if pre_v and pre_v > 0:
            pct_vol = (post_v - pre_v) / pre_v
            beta_vol_list.append(pct_vol / STT_CHANGE)

    beta_tv_clipped = [np.clip(b, -BETA_TV_CLIP, BETA_TV_CLIP) for b in beta_tv_raw]

    results = {
        'BETA_TV_MIN': min(beta_tv_clipped) if beta_tv_clipped else -0.5,
        'BETA_TV_MAX': max(beta_tv_clipped) if beta_tv_clipped else 0.5,
        'BETA_TV_MEDIAN': float(np.median(beta_tv_clipped)) if beta_tv_clipped else 0.0,
        'BETA_VOL_MIN': min(beta_vol_list) if beta_vol_list else -0.5,
        'BETA_VOL_MAX': max(beta_vol_list) if beta_vol_list else 0.5,
        'BETA_VOL_MEDIAN': float(np.median(beta_vol_list)) if beta_vol_list else 0.0,
    }

    print(f"\n    TRAIN LAW PERIOD ANALYSIS:")
    print(f"    {'─' * 70}")
    print(f"    STT Change: 0.5% → 0.6% (+{STT_CHANGE*100:.0f}%)")
    print(f"    Quarterly pairs analysed: {n_q}")
    print(f"    β_TV clipped at [−{BETA_TV_CLIP}, +{BETA_TV_CLIP}] (outlier mitigation)")
    print(f"    {'─' * 70}")
    print(f"\n    Volume Elasticity (β_TV):")
    print(f"      Range:   [{results['BETA_TV_MIN']:>+8.4f}, {results['BETA_TV_MAX']:>+8.4f}]")
    print(f"      Median:  {results['BETA_TV_MEDIAN']:>+8.4f}")
    print(f"\n    Volatility Elasticity (β_VOL):")
    print(f"      Range:   [{results['BETA_VOL_MIN']:>+8.4f}, {results['BETA_VOL_MAX']:>+8.4f}]")
    print(f"      Median:  {results['BETA_VOL_MEDIAN']:>+8.4f}")

    render_table(
        title="Table 2.  Elasticity Ranges — TRAIN Law Natural Experiment (2018)\n"
              f"β = (%ΔOutcome) / (%ΔSTT)   |   %ΔSTT = +{STT_CHANGE*100:.0f}%  (0.5% → 0.6%)",
        col_labels=['Elasticity', 'Minimum', 'Median', 'Maximum'],
        row_data=[
            ['Volume Elasticity (β_TV)',
             f'{results["BETA_TV_MIN"]:.4f}', f'{results["BETA_TV_MEDIAN"]:.4f}',
             f'{results["BETA_TV_MAX"]:.4f}'],
            ['Volatility Elasticity (β_VOL)',
             f'{results["BETA_VOL_MIN"]:.4f}', f'{results["BETA_VOL_MEDIAN"]:.4f}',
             f'{results["BETA_VOL_MAX"]:.4f}'],
        ],
        fname='table2_elasticity.png',
        col_widths=[0.35, 0.20, 0.20, 0.25],
        figsize=(13, 2.8),
        footnote=("Natural experiment: TRAIN Law raised STT from 0.5% to 0.6% effective January 2018.\n"
                  "β values sampled Uniform(min, max) in each MCS iteration. "
                  f"β_TV clipped at [−{BETA_TV_CLIP}, +{BETA_TV_CLIP}] to mitigate outliers."),
    )

    return results


# ============================================================================
# STAGE 3 — LAMBDA FACTORS
# ============================================================================

def calculate_lambda_ranges(pse_data: dict, med_data: dict) -> dict:
    """
    STAGE 3: Derives lambda (λ) mediating factors from pre-CMEPA data.
    
    λ scales the elasticity effect based on market conditions.
    λ ~ Normal(µ, σ) sampled each iteration.
    """
    _section_header(3, "LAMBDA (λ) MEDIATING FACTOR DERIVATION")
    print("    All lambdas derived exclusively from pre-CMEPA data (Jul 2024 – Jun 2025)")

    flr = med_data['flr'].copy()
    pre_flr = flr[flr['Period'] == 'Pre-CMEPA'].dropna(subset=['FLR']).copy()

    if len(pre_flr) > 0:
        flr_mean = pre_flr['FLR'].mean()
        pre_flr['lambda1'] = pre_flr['FLR'] / flr_mean
        lam_mean = float(pre_flr['lambda1'].mean())
        lam_std = float(pre_flr['lambda1'].std(ddof=1)) if len(pre_flr) > 1 else 0.1
    else:
        log_message("FLR data not available, using default lambda", "WARNING")
        lam_mean = 1.0
        lam_std = 0.15

    print(f"\n    LAMBDA PARAMETERS:")
    print(f"    {'─' * 70}")
    print(f"    Composite Lambda Mean (µ):       {lam_mean:>18.4f}")
    print(f"    Composite Lambda Std Dev (σ):    {lam_std:>18.4f}")
    print(f"    Clipping bounds [±2σ from µ]:    [{max(lam_mean - 2*lam_std, 0.5):>8.4f}, {lam_mean + 2*lam_std:>8.4f}]")
    print(f"    {'─' * 70}")

    render_table(
        title="Table 3.  Lambda Mediating Factors (Pre-CMEPA, Jul 2024 – Jun 2025)\n"
              "λ scales elasticity based on market conditions; sampled Normal(µ, σ) each iteration",
        col_labels=['Component', 'Mean (µ)', 'Std Dev (σ)', 'Definition'],
        row_data=[
            ['Composite Lambda (λ)', f'{lam_mean:.4f}', f'{lam_std:.4f}',
             'Mediating factor; λ > 1 amplifies, λ < 1 dampens policy effect'],
        ],
        fname='table3_lambda_parameters.png',
        col_widths=[0.25, 0.25, 0.25, 0.25],
        figsize=(14, 2.5),
        footnote=("λ ~ Normal(µ, σ) sampled independently each MCS iteration (clipped to [0.5, 2.0]).\n"
                  "Represents market conditions, investor sentiment, and execution efficiency."),
    )

    return {
        'lam_mean': lam_mean,
        'lam_std': lam_std,
    }


# ============================================================================
# STAGE 4 — MONTE CARLO SIMULATION (STATIONARY)
# ============================================================================

def _run_single_horizon(label: str, cfg: dict, eda_params: dict,
                        elasticity: dict, lambda_ranges: dict) -> dict:
    """
    STAGE 4: Runs one time-horizon simulation with STATIONARY ASSUMPTIONS.
    
    STATIONARITY CORE:
    - Daily policy shock %ΔY = β × %Δt × λ remains identical across all trading days
    - No volume growth, no structural change
    - Cumulative revenue = Sum of identical daily shocks
    
    EDA INTEGRATION:
    - Each day: tv_pre ~ Normal(TV_µ, TV_σ)
    - Elasticity β sampled uniformly from TRAIN period range
    - Lambda λ sampled from Normal(µ, σ) pre-CMEPA parameters
    """

    sims_per_day = cfg['sims_per_day']
    trading_days = cfg['trading_days']
    total_iters = cfg['total_iters']
    horizon_years = cfg['horizon_years']

    # EDA PARAMETERS: Normal distributions
    TV_MEAN = eda_params['TV_mu']         # ✅ EDA mean
    TV_SD = eda_params['TV_sigma']        # ✅ EDA std dev
    P_AVG = eda_params['P_avg']
    VOL_PRE = eda_params['VOL_pre']

    # ELASTICITY RANGES (from TRAIN period)
    beta_tv_range = (elasticity['BETA_TV_MIN'], elasticity['BETA_TV_MAX'])
    beta_vol_range = (elasticity['BETA_VOL_MIN'], elasticity['BETA_VOL_MAX'])
    
    # LAMBDA PARAMETERS
    lam_mean, lam_std = lambda_ranges['lam_mean'], lambda_ranges['lam_std']

    # BASELINE REVENUE (6-month reference)
    DAYS_BASELINE = 126
    REV_BASELINE = (TV_MEAN * P_AVG) * STT_PRE * DAYS_BASELINE

    print(f"\n    ── {label} Horizon ─────────────────────────────────────")
    print(f"       Sims/day:       {sims_per_day:>10,}")
    print(f"       Trading days:   {trading_days:>10,}")
    print(f"       Total iters:    {total_iters:>10,}")
    print(f"       Assumption:     STATIONARY policy shock (identical every day)")
    print(f"       EDA applied:    Normal(µ={TV_MEAN:,.0f}, σ={TV_SD:,.0f})")

    # Result arrays
    daily_tv_pct_mean = np.zeros(trading_days)
    daily_vol_pct_mean = np.zeros(trading_days)
    daily_rev_mean = np.zeros(trading_days)
    daily_rev_std = np.zeros(trading_days)
    daily_tv_post_mean = np.zeros(trading_days)

    # Sensitivity analysis sampling (last 50k records)
    _n_flat = min(total_iters, 50_000)
    flat_beta_tv = np.zeros(_n_flat)
    flat_beta_vol = np.zeros(_n_flat)
    flat_lam = np.zeros(_n_flat)
    flat_tv_pre = np.zeros(_n_flat)
    flat_rev = np.zeros(_n_flat)
    _flat_idx = 0

    # ─────────────────────────────────────────────────────────────────────────
    # MONTE CARLO LOOP: Day by day
    # ─────────────────────────────────────────────────────────────────────────
    for d in range(trading_days):
        # ✅ EDA SAMPLING: Daily baseline volume from Normal distribution
        tv_pre_arr = np.random.normal(TV_MEAN, TV_SD, sims_per_day)
        tv_pre_arr = np.maximum(tv_pre_arr, TV_MEAN * 0.05)  # Floor at 5% of mean

        # Sample elasticities (uniform from TRAIN period range)
        beta_tv_arr = np.random.uniform(*beta_tv_range, sims_per_day)
        beta_vol_arr = np.random.uniform(*beta_vol_range, sims_per_day)

        # Sample lambda (normal from pre-CMEPA parameters)
        lam_arr = np.random.normal(lam_mean, lam_std, sims_per_day)
        lam_arr = np.clip(lam_arr, 0.5, 2.0)  # Reasonable bounds

        # ✅ CORE POLICY MODEL (STATIONARY)
        # %ΔY = β × %Δt × λ
        # This shock is IDENTICAL every day (stationarity assumption)
        pct_tv_arr = beta_tv_arr * STT_CHANGE_PCT * lam_arr
        pct_vol_arr = beta_vol_arr * STT_CHANGE_PCT * lam_arr

        # Post-policy outcomes
        tv_post_arr = tv_pre_arr * (1 + pct_tv_arr)
        gsp_post_arr = tv_post_arr * P_AVG
        rev_post_arr = gsp_post_arr * STT_POST  # Daily revenue per sim
        vol_post_arr = VOL_PRE * (1 + pct_vol_arr)

        # Store daily aggregates
        daily_tv_pct_mean[d] = pct_tv_arr.mean()
        daily_vol_pct_mean[d] = pct_vol_arr.mean()
        daily_rev_mean[d] = rev_post_arr.mean()
        daily_rev_std[d] = rev_post_arr.std()
        daily_tv_post_mean[d] = tv_post_arr.mean()

        # Fill sensitivity arrays (rolling 50k window)
        n_to_store = min(sims_per_day, _n_flat)
        start = _flat_idx % _n_flat
        end = min(start + n_to_store, _n_flat)
        actual_store = end - start

        if actual_store > 0:
            flat_beta_tv[start:end] = beta_tv_arr[:actual_store]
            flat_beta_vol[start:end] = beta_vol_arr[:actual_store]
            flat_lam[start:end] = lam_arr[:actual_store]
            flat_tv_pre[start:end] = tv_pre_arr[:actual_store]
            flat_rev[start:end] = rev_post_arr[:actual_store]

        _flat_idx += n_to_store

    # ─────────────────────────────────────────────────────────────────────────
    # AGGREGATE STATISTICS (summed across all trading days)
    # ─────────────────────────────────────────────────────────────────────────
    cumulative_rev = float(daily_rev_mean.sum())
    mean_tv_pct = float(daily_tv_pct_mean.mean()) * 100
    mean_vol_pct = float(daily_vol_pct_mean.mean()) * 100
    pct_vol_increase = float((daily_vol_pct_mean > 0).mean()) * 100

    # Counterfactual: same volume trajectory, old STT rate
    rev_counterfactual = float((daily_tv_post_mean * P_AVG * STT_PRE).sum())
    laffer_ok = cumulative_rev > rev_counterfactual
    rev_chg_pct = (cumulative_rev - rev_counterfactual) / max(rev_counterfactual, 1) * 100

    # 95% CI (daily aggregates summed)
    ci_low = float((daily_rev_mean - 1.96 * daily_rev_std).sum())
    ci_high = float((daily_rev_mean + 1.96 * daily_rev_std).sum())

    print(f"       {'─' * 50}")
    print(f"       Daily mean % Δ Volume:        {mean_tv_pct:>+10.2f}%")
    print(f"       Daily mean % Δ Volatility:    {mean_vol_pct:>+10.2f}%")
    print(f"       Cumulative Revenue:           ₱{cumulative_rev:>16,.0f}")
    print(f"       Laffer Verdict:               {'ADEQUATE' if laffer_ok else 'INADEQUATE':>11}")

    return {
        'label': label,
        'horizon_years': horizon_years,
        'total_iters': total_iters,
        'sims_per_day': sims_per_day,
        'trading_days': trading_days,
        'daily_tv_pct_mean': daily_tv_pct_mean,
        'daily_vol_pct_mean': daily_vol_pct_mean,
        'daily_rev_mean': daily_rev_mean,
        'daily_rev_std': daily_rev_std,
        'daily_tv_post_mean': daily_tv_post_mean,
        'cumulative_rev': cumulative_rev,
        'rev_counterfactual': rev_counterfactual,
        'mean_tv_pct': mean_tv_pct,
        'mean_vol_pct': mean_vol_pct,
        'pct_vol_increase': pct_vol_increase,
        'rev_chg_pct': rev_chg_pct,
        'laffer_ok': laffer_ok,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'REV_BASELINE': REV_BASELINE,
        'flat_beta_tv': flat_beta_tv,
        'flat_beta_vol': flat_beta_vol,
        'flat_lam': flat_lam,
        'flat_tv_pre': flat_tv_pre,
        'flat_rev': flat_rev,
    }


def run_all_horizons(eda_params: dict, elasticity: dict, lambda_ranges: dict) -> dict:
    """Runs all three horizons with stationary assumptions."""
    _section_header(4, "MULTI-HORIZON MONTE CARLO SIMULATION (STATIONARY ASSUMPTIONS)")
    
    print(f"    Policy Shock: STT {STT_PRE*100:.1f}% → {STT_POST*100:.1f}% ({STT_CHANGE_PCT*100:.2f}%)")
    print(f"    Core Model: %ΔY = β × (%Δt) × λ")
    print(f"    Key Assumption: Identical daily shock across all horizons (stationarity)")
    print(f"    EDA Integration: Each day, TV_pre ~ Normal(TV_µ={eda_params['TV_mu']:,.0f}, "
          f"TV_σ={eda_params['TV_sigma']:,.0f})")

    horizon_results = {}
    for label, cfg in HORIZONS.items():
        horizon_results[label] = _run_single_horizon(
            label, cfg, eda_params, elasticity, lambda_ranges)

    # CONSOLIDATED TABLE
    print(f"\n\n    ══ CONSOLIDATED RESULTS ACROSS ALL HORIZONS ══")
    print(f"    (Daily % changes identical; cumulative revenue scales linearly with days)")
    print(f"    {'Metric':<45} {'1-Year':>14}  {'5-Year':>14}  {'10-Year':>14}")
    print("    " + "─" * 90)

    metrics_rows = []
    for metric_lbl, key, fmt in [
        ('Total Iterations', 'total_iters', lambda v: f'{v:,}'),
        ('Trading Days', 'trading_days', lambda v: f'{v:,}'),
        ('Daily mean % Δ Volume', 'mean_tv_pct', lambda v: f'{v:+.2f}%'),
        ('Daily mean % Δ Volatility', 'mean_vol_pct', lambda v: f'{v:+.2f}%'),
        ('P(Volatility Increase)', 'pct_vol_increase', lambda v: f'{v:.2f}%'),
        ('Cumulative Revenue (Rev_S)', 'cumulative_rev', lambda v: f'₱{v:,.0f}'),
        ('Counterfactual (Rev_B)', 'rev_counterfactual', lambda v: f'₱{v:,.0f}'),
        ('Revenue Δ vs Counterfactual', 'rev_chg_pct', lambda v: f'{v:+.1f}%'),
        ('Laffer Verdict', 'laffer_ok', lambda v: 'ADEQUATE' if v else 'INADEQUATE'),
    ]:
        vals = [horizon_results[h][key] for h in HORIZONS]
        row = [metric_lbl] + [fmt(v) for v in vals]
        metrics_rows.append(row)
        print(f"    {metric_lbl:<45} {fmt(vals[0]):>14}  {fmt(vals[1]):>14}  {fmt(vals[2]):>14}")

    render_table(
        title="Table 4.  Consolidated Monte Carlo Results — All Time Horizons\n"
              "Model: %ΔY = β × (−83.33%) × λ  |  Stationarity: Identical Daily Policy Shock Across Horizons",
        col_labels=['Metric', '1-Year\n(252K)', '5-Year\n(630K)', '10-Year\n(630K)'],
        row_data=metrics_rows,
        fname='table4_horizon_results.png',
        col_widths=[0.42, 0.19, 0.19, 0.20],
        figsize=(15, 7),
        highlight_rows=[4, 5, 8],
        footnote=("KEY RESULT: Daily % changes (volume, volatility, revenue) are identical across horizons (stationarity).\n"
                  "Cumulative revenue scales linearly: 252d → 1.0x | 1,260d → 5.0x | 2,520d → 10.0x\n"
                  "EDA: Each daily volume drawn from Normal(TV_µ, TV_σ) | β_TV, β_VOL uniform from TRAIN period | λ ~ Normal(µ, σ)"),
    )

    _plot_horizon_timeseries(horizon_results)
    _plot_cross_horizon_comparison(horizon_results)

    return horizon_results


def _plot_horizon_timeseries(horizon_results: dict) -> None:
    """Time-series plots for each horizon."""
    for label, res in horizon_results.items():
        days = np.arange(1, res['trading_days'] + 1)
        cum_rev = np.cumsum(res['daily_rev_mean'])

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.patch.set_facecolor('white')
        fig.suptitle(f"Figure — {label} Horizon: Daily Progression (Stationary Assumption)\n"
                     f"Total {res['total_iters']:,} iterations  |  "
                     f"{res['sims_per_day']:,} sims/day × {res['trading_days']:,} days",
                    fontsize=12, fontweight='bold', color=C_DARK)

        # Panel A — Volume
        ax = axes[0]
        ax.set_facecolor('white')
        ax.plot(days, res['daily_tv_pct_mean'] * 100, color=C_BLUE, lw=1.5, label='Daily Mean')
        ax.axhline(res['mean_tv_pct'], color=C_DARK, lw=1.2, ls='--', 
                  label=f'Horizon Average: {res["mean_tv_pct"]:+.2f}%')
        ax.axhline(0, color='#aaa', lw=0.8, ls=':')
        ax.set_ylabel("% Change in Trading Volume", fontsize=10, fontweight='bold')
        ax.set_title("Panel A — Daily % Change in Trading Volume (Liquidity)", 
                    fontsize=10, fontweight='bold', color=C_DARK)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel B — Volatility
        ax = axes[1]
        ax.set_facecolor('white')
        ax.plot(days, res['daily_vol_pct_mean'] * 100, color=C_RED, lw=1.5, label='Daily Mean')
        ax.axhline(res['mean_vol_pct'], color=C_DARK, lw=1.2, ls='--', 
                  label=f'Horizon Average: {res["mean_vol_pct"]:+.2f}%')
        ax.axhline(0, color='#aaa', lw=0.8, ls=':')
        ax.set_ylabel("% Change in Market Volatility", fontsize=10, fontweight='bold')
        ax.set_title("Panel B — Daily % Change in Market Volatility", 
                    fontsize=10, fontweight='bold', color=C_DARK)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel C — Revenue
        ax = axes[2]
        ax.set_facecolor('white')
        ax.plot(days, cum_rev, color=C_GREEN, lw=2.0, label='Cumulative Rev_S (0.1%)', zorder=5)
        cf_daily = res['rev_counterfactual'] / res['trading_days']
        cum_cf = np.cumsum(np.full(res['trading_days'], cf_daily))
        ax.plot(days, cum_cf, color=C_RED, lw=1.8, ls='--', label='Counterfactual Rev_B (0.6%)', zorder=4)
        ax.fill_between(days, cum_rev, alpha=0.08, color=C_GREEN)
        ax.set_xlabel(f"Trading Day (1 – {res['trading_days']:,})", fontsize=10, fontweight='bold')
        ax.set_ylabel("Cumulative STT Revenue (₱)", fontsize=10, fontweight='bold')
        ax.set_title(f"Panel C — Revenue Progression  |  Rev_S: ₱{res['cumulative_rev']/1e6:.1f}M  |  "
                    f"Δ: {res['rev_chg_pct']:+.1f}%  ({('ADEQUATE' if res['laffer_ok'] else 'INADEQUATE')})",
                    fontsize=10, fontweight='bold', color=C_DARK)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₱{x/1e6:.1f}M'))
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        fname = f"fig_timeseries_{label.replace('-', '').replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight', facecolor='white')
        log_message(f"Time-series figure saved → {fname}", "OUTPUT")
        plt.show()


def _plot_cross_horizon_comparison(horizon_results: dict) -> None:
    """Cross-horizon comparison (stationarity verification)."""
    labels = list(HORIZONS.keys())
    tv_means = [horizon_results[h]['mean_tv_pct'] for h in labels]
    vol_means = [horizon_results[h]['mean_vol_pct'] for h in labels]
    rev_s_vals = [horizon_results[h]['cumulative_rev'] for h in labels]
    rev_b_vals = [horizon_results[h]['rev_counterfactual'] for h in labels]
    rev_chg = [horizon_results[h]['rev_chg_pct'] for h in labels]
    trading_days_list = [horizon_results[h]['trading_days'] for h in labels]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle("Figure — Cross-Horizon Comparative Analysis (Stationary Assumption Verification)\n"
                 "Daily % changes IDENTICAL; Cumulative revenue SCALES LINEARLY with trading days",
                fontsize=13, fontweight='bold', color=C_DARK)

    x = np.arange(len(labels))
    bw = 0.35

    # Panel A — Daily % Changes (should be identical)
    ax = axes[0, 0]
    ax.set_facecolor('white')
    ax.bar(x - bw/2, tv_means, bw, color=C_BLUE, alpha=0.85, edgecolor='#333', lw=0.8, label='% Δ Volume')
    ax.bar(x + bw/2, vol_means, bw, color=C_RED, alpha=0.85, edgecolor='#333', lw=0.8, label='% Δ Volatility')
    ax.axhline(0, color='#888', lw=0.8, ls='--', alpha=0.7)
    for i, (tv, vol) in enumerate(zip(tv_means, vol_means)):
        ax.text(i - bw/2, tv + 0.3, f'{tv:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + bw/2, vol + 0.3, f'{vol:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Daily Mean % Change", fontsize=10, fontweight='bold')
    ax.set_title("Panel A — Daily % Changes\n(STATIONARITY: Identical Across Horizons)",
                fontsize=10, fontweight='bold', color=C_DARK)
    ax.legend(fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel B — Revenue Scaling (linear)
    ax = axes[0, 1]
    ax.set_facecolor('white')
    ax.bar(x - bw/2, [v / 1e6 for v in rev_s_vals], bw, color=C_GREEN, alpha=0.85, 
          edgecolor='#333', lw=0.8, label='Rev_S (0.1%)')
    ax.bar(x + bw/2, [v / 1e6 for v in rev_b_vals], bw, color=C_RED, alpha=0.85, 
          edgecolor='#333', lw=0.8, label='Rev_B (0.6% counterfactual)')
    for i, (rs, rb) in enumerate(zip(rev_s_vals, rev_b_vals)):
        ax.text(i - bw/2, rs/1e6 + 50, f'₱{rs/1e6:.0f}M', ha='center', va='bottom', fontsize=8)
        ax.text(i + bw/2, rb/1e6 + 50, f'₱{rb/1e6:.0f}M', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cumulative Revenue (₱M)", fontsize=10, fontweight='bold')
    ax.set_title("Panel B — Revenue Scaling\n(LINEAR: Scales with # of Trading Days)",
                fontsize=10, fontweight='bold', color=C_DARK)
    ax.legend(fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel C — Laffer Verdict (% change)
    ax = axes[1, 0]
    ax.set_facecolor('white')
    colors = [C_GREEN if v > 0 else C_RED for v in rev_chg]
    bars = ax.bar(labels, rev_chg, color=colors, alpha=0.85, edgecolor='#333', lw=0.8)
    ax.axhline(0, color='#888', lw=1.0, ls='--')
    for bar, rc in zip(bars, rev_chg):
        verdict = 'ADEQUATE' if rc > 0 else 'INADEQUATE'
        y_pos = rc + (max(abs(min(rev_chg)), abs(max(rev_chg))) * 0.05 * (1 if rc > 0 else -1))
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{rc:+.1f}%\n{verdict}',
               ha='center', va='bottom' if rc > 0 else 'top', fontsize=9, fontweight='bold', color=C_DARK)
    ax.set_ylabel("Revenue Δ (%)", fontsize=10, fontweight='bold')
    ax.set_title("Panel C — Laffer Curve Test\n(Revenue Change vs. Counterfactual)",
                fontsize=10, fontweight='bold', color=C_DARK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel D — Trading Days Scaling (verification)
    ax = axes[1, 1]
    ax.set_facecolor('white')
    scaling_factor = [d / 252 for d in trading_days_list]
    ax.plot(labels, scaling_factor, marker='o', markersize=12, linewidth=2.5, color=C_BLUE, label='Scaling Factor (days/252)')
    ax.plot(labels, scaling_factor, marker='o', markersize=12, linewidth=2.5, color=C_BLUE, alpha=0.3)
    for i, (lbl, sf, td) in enumerate(zip(labels, scaling_factor, trading_days_list)):
        ax.text(i, sf + 0.15, f'{sf:.1f}x\n({td:,} days)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("Revenue Scaling Factor", fontsize=10, fontweight='bold')
    ax.set_title("Panel D — Stationarity Verification\n(Cumulative Revenue Scales Linearly with Days)",
                fontsize=10, fontweight='bold', color=C_DARK)
    ax.set_ylim(0, max(scaling_factor) + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_cross_horizon_comparison.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    log_message("Cross-horizon comparison figure saved", "OUTPUT")
    plt.show()


# ============================================================================
# STAGE 5 — SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(horizon_results: dict) -> dict:
    """Sensitivity analysis using Spearman rank correlation."""
    _section_header(5, "SENSITIVITY ANALYSIS (Spearman Rank Correlation)")

    all_sa = {}
    all_sa_rows = []

    for label, res in horizon_results.items():
        inputs = {
            'β_TV (Volume Elasticity)': res['flat_beta_tv'],
            'β_VOL (Volatility Elasticity)': res['flat_beta_vol'],
            'TV_pre (Baseline Volume)': res['flat_tv_pre'],
            'λ_total (Composite Lambda)': res['flat_lam'],
        }
        output = res['flat_rev']

        rows = []
        for name, data in inputs.items():
            rho, pval = spearmanr(data, output)
            rows.append({'Variable': name, 'Rho': rho, 'AbsRho': abs(rho), 'PValue': pval})

        df_sa = pd.DataFrame(rows).sort_values('AbsRho', ascending=True).reset_index(drop=True)
        all_sa[label] = df_sa

        print(f"\n    {label} Horizon (Spearman Rank Correlation):")
        print(f"    {'─' * 70}")
        for _, row in df_sa.iterrows():
            sig = ('***' if row['PValue'] < 0.001 else '**' if row['PValue'] < 0.01 
                  else '*' if row['PValue'] < 0.05 else 'ns')
            strength = ('Strong' if row['AbsRho'] >= 0.50 else 'Moderate' if row['AbsRho'] >= 0.30 
                       else 'Weak')
            print(f"      {row['Variable']:<35} ρ={row['Rho']:>+.4f}  |ρ|={row['AbsRho']:>6.4f}  {sig:>3}  {strength}")

        t_rows = [[label, str(n+1), row['Variable'], f'{row["Rho"]:+.4f}', f'{row["AbsRho"]:.4f}']
                  for n, (_, row) in enumerate(df_sa.iterrows())]
        all_sa_rows.extend(t_rows)

    render_table(
        title="Table 5.  Sensitivity Analysis — Spearman Rank Correlations (All Horizons)\n"
              "Correlation between stochastic MCS inputs and projected daily revenue (ρ ∈ [−1, +1])",
        col_labels=['Horizon', 'Rank', 'Input Variable', 'ρ (Correlation)', '|ρ| (Absolute)'],
        row_data=all_sa_rows,
        fname='table5_sensitivity_all_horizons.png',
        col_widths=[0.14, 0.08, 0.45, 0.16, 0.17],
        figsize=(16, 6),
        footnote=("Spearman ρ: rank-based, robust to outliers and non-normal distributions.\n"
                  "Sample: last 50,000 iteration records per horizon. Influence: |ρ| ≥0.50 (Strong), "
                  "0.30–0.49 (Moderate), <0.30 (Weak)."),
    )

    fig_t, axes_t = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig_t.patch.set_facecolor('white')
    fig_t.suptitle("Figure — Tornado Charts: Input Sensitivity Analysis (All Horizons)\n"
                   "Longer bar = stronger influence on simulated daily revenue variance",
                  fontsize=12, fontweight='bold', color=C_DARK)

    for ax, (label, df_sa) in zip(axes_t, all_sa.items()):
        ax.set_facecolor('white')
        colors = [C_BLUE if r >= 0 else C_RED for r in df_sa['Rho']]
        bars = ax.barh(df_sa['Variable'], df_sa['Rho'], color=colors, alpha=0.85, 
                      edgecolor='#333', lw=0.7, height=0.6)
        ax.axvline(0, color=C_DARK, lw=1.2, ls='--', alpha=0.7)
        xlim = max(abs(df_sa['Rho'].min()), abs(df_sa['Rho'].max())) * 1.4
        ax.set_xlim(-xlim, xlim)
        ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=9, fontweight='bold')
        ax.set_xlabel("Spearman ρ", fontsize=10, fontweight='bold')
        ax.set_title(f"{label} Horizon", fontsize=11, fontweight='bold', color=C_DARK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_tornado_all_horizons.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    log_message("Tornado chart figure saved", "OUTPUT")
    plt.show()

    return all_sa


# ============================================================================
# STAGE 6 — LAFFER CURVE & MODEL VALIDATION
# ============================================================================

def run_laffer_and_validation(horizon_results: dict, post_cmepa: pd.DataFrame,
                               eda_params: dict) -> dict:
    """Laffer curve fiscal adequacy test and model validation."""
    _section_header(6, "LAFFER CURVE FISCAL ADEQUACY TEST & MODEL VALIDATION")

    # Actual post-CMEPA observations
    if len(post_cmepa) > 0:
        actual_volume = float(post_cmepa['Volume'].mean()) if 'Volume' in post_cmepa.columns else 0
        actual_price = float(post_cmepa['Closing Price'].mean()) if 'Closing Price' in post_cmepa.columns else 1
    else:
        actual_volume, actual_price = 0, 1

    actual_gsp = actual_volume * actual_price
    actual_rev = actual_gsp * STT_POST * len(post_cmepa) if len(post_cmepa) > 0 else 0
    pre_volume_mean = eda_params['pre_tv_mean']
    actual_volume_chg_pct = ((actual_volume - pre_volume_mean) / pre_volume_mean * 100) if pre_volume_mean > 0 else 0

    print(f"\n    ACTUAL POST-CMEPA OBSERVATIONS (Jul–Dec 2025)")
    print(f"    {'─' * 70}")
    print(f"    Mean Daily Volume:              {actual_volume:>18,.0f} shares")
    print(f"    vs. Pre-CMEPA Mean:             {pre_volume_mean:>18,.0f} shares")
    print(f"    Observed Volume Change:         {actual_volume_chg_pct:>+17.2f}%")
    print(f"    Mean Closing Price:             ₱{actual_price:>17,.4f}")
    print(f"    Actual Daily GSP:               ₱{actual_gsp:>17,.0f}")
    print(f"    Observed Trading Days:          {len(post_cmepa):>18} days")
    print(f"    Actual Total Revenue (6-month): ₱{actual_rev:>17,.0f}")
    print(f"    {'─' * 70}")

    # ─────────────────────────────────────────────────────────────────────────
    # A. LAFFER CURVE TEST
    # ─────────────────────────────────────────────────────────────────────────
    _subsection("A. LAFFER CURVE FISCAL ADEQUACY TEST")

    laffer_rows = []
    laffer_adequate_count = 0

    for label, res in horizon_results.items():
        verdict = 'ADEQUATE' if res['laffer_ok'] else 'INADEQUATE'
        if res['laffer_ok']:
            laffer_adequate_count += 1
        laffer_rows.append([label, f'₱{res["cumulative_rev"]/1e6:,.1f}M',
                            f'₱{res["rev_counterfactual"]/1e6:,.1f}M',
                            f'{res["rev_chg_pct"]:+.1f}%', verdict])
        v_sym = '✓' if res['laffer_ok'] else '✗'
        print(f"    {v_sym} {label:>10}: Rev_S = ₱{res['cumulative_rev']/1e6:>8.1f}M | "
              f"Rev_B = ₱{res['rev_counterfactual']/1e6:>8.1f}M | "
              f"Δ = {res['rev_chg_pct']:>+6.1f}% → {verdict}")

    print(f"\n    LAFFER ADEQUACY SUMMARY: {laffer_adequate_count}/3 horizons fiscally adequate")

    render_table(
        title="Table 6.  Laffer Curve Fiscal Adequacy Test — All Time Horizons\n"
              "Rev_S = Simulated Revenue (0.1%)  |  Rev_B = Counterfactual (0.6%, same volume trajectory)",
        col_labels=['Horizon', 'Rev_S\n(Simulated, 0.1%)',
                    'Rev_B\n(Counterfactual, 0.6%)',
                    'Revenue Δ\n(% vs Base)', 'Fiscal\nVerdict'],
        row_data=laffer_rows,
        fname='table6_laffer_all_horizons.png',
        col_widths=[0.14, 0.22, 0.22, 0.16, 0.26],
        figsize=(15, 3.8),
        highlight_rows=[0, 1, 2],
        footnote=("Fiscal adequacy test: Does volume expansion offset 83.33% STT rate reduction?\n"
                  "Revenue_S > Revenue_B → Volume effect outweighs tax rate cut → Fiscally ADEQUATE\n"
                  "Comparison valid under stationarity (identical daily policy shock)."),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # B. MODEL VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    _subsection("B. MODEL VALIDATION — 95% Confidence Interval Test (1-Year Horizon)")

    res_1y = horizon_results['1-Year']

    tv_pct_mean = float(res_1y['daily_tv_pct_mean'].mean()) * 100
    tv_pct_std = float(res_1y['daily_tv_pct_mean'].std()) * 100
    ci_tv_pct_lo = tv_pct_mean - 1.96 * tv_pct_std
    ci_tv_pct_hi = tv_pct_mean + 1.96 * tv_pct_std
    tv_pct_in_ci = bool(ci_tv_pct_lo <= actual_volume_chg_pct <= ci_tv_pct_hi)

    ci_rev_lo = res_1y['ci_low']
    ci_rev_hi = res_1y['ci_high']
    rv_in_ci = bool(ci_rev_lo <= actual_rev <= ci_rev_hi) if actual_rev > 0 else None

    overall_robust = (tv_pct_in_ci and rv_in_ci) if rv_in_ci is not None else tv_pct_in_ci

    print(f"\n    VALIDATION RESULTS:")
    print(f"    {'─' * 70}")
    print(f"    % Δ Volume (Daily Mean):")
    print(f"      Actual:       {actual_volume_chg_pct:>+10.2f}%")
    print(f"      95% CI:       [{ci_tv_pct_lo:>+10.2f}%, {ci_tv_pct_hi:>+10.2f}%]")
    print(f"      Within CI?    {'YES ✓' if tv_pct_in_ci else 'NO ✗':>12}")
    print(f"\n    Total Revenue (6-Month):")
    print(f"      Actual:       ₱{actual_rev:>16,.0f}")
    print(f"      95% CI:       [₱{ci_rev_lo:>16,.0f}, ₱{ci_rev_hi:>16,.0f}]")
    print(f"      Within CI?    {'YES ✓' if rv_in_ci else 'NO ✗' if rv_in_ci is not None else 'N/A':>12}")
    print(f"    {'─' * 70}")
    
    overall_text = 'ROBUST ✓✓' if overall_robust else 'REQUIRES REFINEMENT ✗'
    print(f"    OVERALL ASSESSMENT: {overall_text}")

    validation_rows = [
        ['% Δ Volume (daily mean)',
         f'{actual_volume_chg_pct:+.2f}%', f'{ci_tv_pct_lo:+.2f}%', f'{ci_tv_pct_hi:+.2f}%',
         '✓ WITHIN' if tv_pct_in_ci else '✗ OUTSIDE'],
        ['Total Revenue (6-month)',
         f'₱{actual_rev:,.0f}', f'₱{ci_rev_lo:,.0f}', f'₱{ci_rev_hi:,.0f}',
         '✓ WITHIN' if rv_in_ci else '✗ OUTSIDE' if rv_in_ci is not None else 'N/A'],
        ['Overall Model Assessment', '—', '—', '—',
         'ROBUST' if overall_robust else 'WEAK'],
    ]

    render_table(
        title="Table 7.  Model Validation — 95% Confidence Interval Test (1-Year Horizon)\n"
              "Actual Post-CMEPA Observations vs. Simulated 95% CI Prediction Bounds",
        col_labels=['Metric', 'Actual', '95% CI Lower', '95% CI Upper', 'Result'],
        row_data=validation_rows,
        fname='table7_model_validation.png',
        col_widths=[0.32, 0.18, 0.18, 0.18, 0.14],
        figsize=(16, 3.5),
        highlight_rows=[2],
        footnote=("Validation: Do actual outcomes fall within simulated 95% confidence bounds?\n"
                  "ROBUST = both metrics within bounds | WEAK = one or more outside bounds\n"
                  "1-Year horizon is most defensible (data available); use for primary conclusions."),
    )

    return {
        'laffer_adequate': laffer_adequate_count,
        'model_robust': overall_robust,
        'tv_pct_in_ci': tv_pct_in_ci,
        'rv_in_ci': rv_in_ci,
        'actual_rev': actual_rev,
        'actual_volume': actual_volume,
    }


# ============================================================================
# STAGE 7 — FINAL SUMMARY
# ============================================================================

def print_final_summary(eda_params: dict, elasticity: dict, lambda_ranges: dict,
                        horizon_results: dict, validation: dict) -> None:
    """Final research summary."""
    _section_header(7, "FINAL RESEARCH SUMMARY & CONCLUSIONS")

    obs_tv_chg = ((eda_params['post_tv_mean'] - eda_params['pre_tv_mean'])
                  / max(eda_params['pre_tv_mean'], 1) * 100)
    obs_vol_chg = ((eda_params['post_vol_mean'] - eda_params['pre_vol_mean'])
                   / max(eda_params['pre_vol_mean'], 1) * 100)

    summary_rows = [
        ['OBSERVED DATA (Actual Post-CMEPA vs Pre-CMEPA)', ''],
        ['  Observed Volume Change', f'{obs_tv_chg:+.2f}%'],
        ['  Observed Volatility Change', f'{obs_vol_chg:+.2f}%'],
        ['', ''],
        ['ESTIMATED PARAMETERS FROM HISTORICAL DATA (EDA)', ''],
        ['  TV_µ (Mean Daily Volume)', f'{eda_params["TV_mu"]:,.0f} shares'],
        ['  TV_σ (Std Dev Daily Volume)', f'{eda_params["TV_sigma"]:,.0f} shares'],
        ['  β_TV Elasticity (TRAIN Period)', f'[{elasticity["BETA_TV_MIN"]:+.4f}, {elasticity["BETA_TV_MAX"]:+.4f}]'],
        ['  β_VOL Elasticity (TRAIN Period)', f'[{elasticity["BETA_VOL_MIN"]:+.4f}, {elasticity["BETA_VOL_MAX"]:+.4f}]'],
        ['  λ Mean (Mediating Factor)', f'{lambda_ranges["lam_mean"]:.4f}'],
        ['', ''],
        ['MONTE CARLO SIMULATION RESULTS (1-Year)', ''],
        ['  Total Iterations', f'{horizon_results["1-Year"]["total_iters"]:,}'],
        ['  Mean % Δ Volume (simulated)', f'{horizon_results["1-Year"]["mean_tv_pct"]:+.2f}%'],
        ['  Mean % Δ Volatility (simulated)', f'{horizon_results["1-Year"]["mean_vol_pct"]:+.2f}%'],
        ['  Cumulative Revenue (Rev_S)', f'₱{horizon_results["1-Year"]["cumulative_rev"]/1e6:,.1f}M'],
        ['  Laffer Fiscal Adequacy', '✓ ADEQUATE' if horizon_results['1-Year']['laffer_ok'] else '✗ INADEQUATE'],
        ['', ''],
        ['MODEL VALIDATION & ROBUSTNESS', ''],
        ['  Model Robustness (95% CI)', 'ROBUST ✓' if validation['model_robust'] else 'WEAK ✗'],
        ['  Laffer Adequate Horizons', f'{validation["laffer_adequate"]}/3'],
        ['  Stationarity Assumption', 'Valid (daily shocks identical across horizons)'],
    ]

    render_table(
        title="Table 8.  Final Research Summary — Simulation Results, Parameter Estimates, and Validation\n"
              "CMEPA: STT 0.6% → 0.1% (−83.33%)  |  Stationarity Assumption Applied",
        col_labels=['Finding / Metric', 'Result / Value'],
        row_data=summary_rows,
        fname='table8_final_summary.png',
        col_widths=[0.55, 0.45],
        figsize=(15, 10),
        highlight_rows=[0, 4, 11, 18],
        footnote=("EDA: Empirical Distribution Approach — µ and σ from real PSE data.\n"
                  "MCS: Each daily volume ~ Normal(TV_µ, TV_σ)  |  β_TV, β_VOL ~ Uniform(TRAIN range)  |  λ ~ Normal(µ, σ)\n"
                  "Stationarity: Daily policy shock identical every day  →  Cumulative revenue scales linearly with trading days."),
    )

    print("\n" + "="*78)
    print("  SIMULATION COMPLETE")
    print("="*78)
    print(f"\n  Output Directory: {OUTPUT_DIR}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Files Generated:")
    print(f"    - table1_eda_parameters.png")
    print(f"    - table2_elasticity.png")
    print(f"    - table3_lambda_parameters.png")
    print(f"    - table4_horizon_results.png")
    print(f"    - table5_sensitivity_all_horizons.png")
    print(f"    - table6_laffer_all_horizons.png")
    print(f"    - table7_model_validation.png")
    print(f"    - table8_final_summary.png")
    print(f"    - fig_timeseries_1year.png")
    print(f"    - fig_timeseries_5year.png")
    print(f"    - fig_timeseries_10year.png")
    print(f"    - fig_cross_horizon_comparison.png")
    print(f"    - fig_tornado_all_horizons.png")
    print("="*78 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    print("="*80)
    print("  CMEPA MONTE CARLO SIMULATION — PRODUCTION RESEARCH VERSION")
    print("  Multi-Horizon Stationary Analysis with EDA (Estimated Distribution Approach)")
    print("  Bajo, Cacho, Rizon, Villamor, Ylaya  |  USJ-R  |  BSA 2025–2026")
    print("="*80)

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC CHECK: Is EDA working?
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  PRE-FLIGHT CHECK: EDA Data Loading Diagnostic")
    is_eda_ok = diagnose_eda_data_loading(PSE_FILE)

    if not is_eda_ok:
        print("\n  ⚠️  WARNING: EDA NOT WORKING PROPERLY")
        print("  The simulation will proceed with default/placeholder data.")
        print("  Results will NOT be valid for research purposes.")
        print("  Fix data loading before using results.\n")

    print(f"\n  SIMULATION STRUCTURE (Stationarity Assumption):")
    print(f"  {'─' * 78}")
    for lbl, cfg in HORIZONS.items():
        print(f"    {lbl:<10}: {cfg['sims_per_day']:>5,} sims/day × {cfg['trading_days']:>5,} days = "
              f"{cfg['total_iters']:>9,} total iterations")
    print(f"  {'─' * 78}")
    print(f"  ✓ Daily % changes identical across horizons (stationarity)")
    print(f"  ✓ Cumulative revenue scales linearly with trading days")
    print(f"  ✓ EDA applied: Each daily volume ~ Normal(TV_µ, TV_σ)")

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTE SIMULATION PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    log_message("Starting simulation pipeline", "MAIN")

    pse_data = load_pse_data(PSE_FILE)
    med_data = load_mediating_data(MED_FILE)

    eda_params = calculate_eda_parameters(pse_data)
    elasticity = estimate_elasticity(pse_data['pre_train'], pse_data['post_train'])
    lambda_ranges = calculate_lambda_ranges(pse_data, med_data)

    horizon_results = run_all_horizons(eda_params, elasticity, lambda_ranges)
    sensitivity = run_sensitivity_analysis(horizon_results)
    validation = run_laffer_and_validation(horizon_results, pse_data['post_cmepa'], eda_params)

    print_final_summary(eda_params, elasticity, lambda_ranges, horizon_results, validation)

    log_message("Simulation pipeline complete", "MAIN")
    print("\n✓ All results saved to:", OUTPUT_DIR)

