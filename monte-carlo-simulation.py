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
MODEL VALIDATION: 95% CI test, distribution comparison, robustness assessment
LAST UPDATED: 2026-04-13
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr, norm, kstest, normaltest
import warnings
import os
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

STT_PRE        = 0.006
STT_POST       = 0.001
STT_CHANGE_PCT = (STT_POST - STT_PRE) / STT_PRE

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

PSE_FILE = "PSE Dataset (1).xlsx"
MED_FILE = "Mediating_Variables Dataset (1).xlsx"
OUTPUT_DIR = "CMEPA-Simulation-Results"

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

    if col_widths:
        for ci, w in enumerate(col_widths):
            for ri in range(n_rows + 1):
                tbl[ri, ci].set_width(w)

    fig.suptitle(title, fontsize=10, fontweight='bold', x=0.5, y=0.99,
                 ha='center', va='top', color=C_DARK, fontfamily='serif')

    if footnote:
        fig.text(0.015, 0.01, footnote, fontsize=7.5, color='#444444',
                 style='italic', va='bottom', fontfamily='serif', wrap=True)

    plt.tight_layout(rect=[0, 0.07 if footnote else 0, 1, 0.96])

    if fname:
        full_path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"    [PNG saved] {fname}")

    plt.show()
    return fig


def _section_header(stage: int, title: str) -> None:
    """Prints formatted section header."""
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"  STAGE {stage} — {title}")
    print(bar)


def _subsection(title: str) -> None:
    """Prints subsection header."""
    print(f"\n  {'─' * 76}")
    print(f"    {title}")
    print(f"  {'─' * 76}")


# ============================================================================
# STAGE 0 — DATA LOADING & EDA
# ============================================================================

def load_pse_data(file: str) -> dict:
    """Loads PSE data from Excel."""

    def _clean(sheet: str, date_lo: str = None, date_hi: str = None) -> pd.DataFrame:
        df = pd.read_excel(file, sheet_name=sheet)
        df.columns = [col.strip() for col in df.columns]

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Closing Price'] = pd.to_numeric(df['Closing Price'], errors='coerce')
        df['Trade Value'] = pd.to_numeric(df['Trade Value'], errors='coerce')
        df['Daily Returns'] = pd.to_numeric(df['Daily Returns'], errors='coerce')
        df['Volume'] = pd.to_numeric(df.get('Volume', pd.Series(dtype=float)), errors='coerce')

        vol_cols = ['SHORT TERM VOLATILITY (5-DAY WINDOW)', 'Volatility', 'VOL', '5-Day Volatility']
        vol_col = None
        for col in vol_cols:
            if col in df.columns:
                vol_col = col
                break

        if vol_col:
            df['Volatility'] = pd.to_numeric(df[vol_col], errors='coerce')
        else:
            df['Volatility'] = 0.015

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

    log_message("PSE Data Loaded", "DATA")

    for lbl, df in [('Pre-TRAIN', pre_train), ('Post-TRAIN', post_train),
                    ('Pre-CMEPA', pre_cmepa), ('Post-CMEPA', post_cmepa)]:
        if len(df):
            print(f"    {lbl:<14}: {len(df):>4} trading days ({df['Date'].min().date()} – {df['Date'].max().date()})")

    return {'pre_train': pre_train, 'post_train': post_train,
            'pre_cmepa': pre_cmepa, 'post_cmepa': post_cmepa}


def load_mediating_data(file: str) -> dict:
    """Loads mediating variables."""
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
        return {'gdp': gdp, 'inflation': inf, 'mkt_pre': mkt_pre, 'mkt_post': mkt_post, 'flr': flr}

    except Exception as e:
        log_message(f"Mediating data error: {e}", "WARNING")
        return {'gdp': pd.DataFrame(), 'inflation': pd.DataFrame(),
                'mkt_pre': pd.DataFrame(), 'mkt_post': pd.DataFrame(), 'flr': pd.DataFrame()}


# ============================================================================
# STAGE 1 — EDA PARAMETER CALCULATION
# ============================================================================

def calculate_eda_parameters(pse_data: dict) -> dict:
    """STAGE 1: Calculates EDA parameters."""
    _section_header(1, "EDA PARAMETER CALCULATION")
    print("    Baseline window: Jul 2024 – Dec 2024")

    pre_full = pse_data['pre_cmepa'].copy()
    for col in ['Daily Returns', 'Closing Price', 'Trade Value', 'Volume', 'Volatility']:
        pre_full[col] = pd.to_numeric(pre_full[col], errors='coerce')

    pre = pre_full[(pre_full['Date'] >= '2024-07-01') &
                   (pre_full['Date'] <= '2024-12-31')].copy()
    if len(pre) == 0:
        pre = pre_full.copy()

    vp = pre.dropna(subset=['Trade Value', 'Volume', 'Closing Price'])

    TV_mu = float(vp['Volume'].mean())
    TV_sigma = float(vp['Volume'].std())
    P_avg = (float(vp['Trade Value'].sum() / vp['Volume'].sum())
             if vp['Volume'].sum() > 0
             else float(vp['Closing Price'].mean()))
    GSP_Mean = TV_mu * P_avg
    VOL_pre = float(pre['Volatility'].dropna().mean())

    pre_tv_mean = float(pse_data['pre_cmepa']['Trade Value'].mean())
    post_tv_mean = float(pse_data['post_cmepa']['Trade Value'].mean())
    pre_vol_mean = float(pse_data['pre_cmepa']['Volatility'].mean())
    post_vol_mean = float(pse_data['post_cmepa']['Volatility'].mean())

    print(f"\n    EDA PARAMETER CALCULATIONS:")
    print(f"    {'─' * 76}")
    print(f"    Parameter                            Result                Unit")
    print(f"    {'─' * 76}")
    print(f"    TV_µ = E[Volume]                     {TV_mu:>30,.2f} shares")
    print(f"    TV_σ = SD[Volume]                    {TV_sigma:>30,.2f} shares")
    print(f"    P_avg = ΣTradeValue / ΣVolume        ₱{P_avg:>29,.4f} per share")
    print(f"    GSP_Mean = TV_µ × P_avg              ₱{GSP_Mean:>29,.2f}")
    print(f"    VOL_pre = E[5-day volatility]        {VOL_pre:>30.6f}")
    print(f"    {'─' * 76}")

    render_table(
        title="Table 1.  EDA Baseline Parameters (Jul 2024 – Dec 2024)\n"
              "These parameters define Normal distributions for Monte Carlo sampling",
        col_labels=['Parameter', 'Symbol', 'Value', 'Calculation', 'Unit'],
        row_data=[
            ['Mean Daily Volume', 'TV_µ', f'{TV_mu:,.2f}', 'mean(Volume)', 'shares'],
            ['Std Dev Volume', 'TV_σ', f'{TV_sigma:,.2f}', 'std(Volume)', 'shares'],
            ['VWAP', 'P_avg', f'₱{P_avg:,.4f}', 'ΣTV/ΣV', '₱/share'],
            ['Mean Daily GSP', 'GSP_µ', f'₱{GSP_Mean:,.2f}', 'TV_µ × P_avg', '₱'],
            ['Pre-CMEPA Volatility', 'VOL_pre', f'{VOL_pre:.6f}', 'mean(5d-vol)', 'σ'],
        ],
        fname='table1_eda_parameters.png',
        col_widths=[0.25, 0.12, 0.20, 0.25, 0.18],
        figsize=(15, 3.8),
        footnote=("EDA: Each MCS iteration samples TV_pre ~ Normal(µ, σ)  |  "
                  "Real data from Jul–Dec 2024 (252 trading days)"),
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
    """STAGE 2: Elasticity from TRAIN Law."""
    _section_header(2, "ELASTICITY ESTIMATION (TRAIN Law Natural Experiment)")

    STT_CHANGE = 0.20

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
    beta_tv_list = []
    beta_vol_list = []

    for i in range(n_q):
        pre_v = pre_q.loc[i, 'volume_mean']
        post_v = post_q.loc[i, 'volume_mean']
        pct_tv = (post_v - pre_v) / pre_v if pre_v > 0 else 0
        beta_tv = pct_tv / STT_CHANGE
        beta_tv_clipped = np.clip(beta_tv, -2.5, 2.5)
        beta_tv_list.append(beta_tv_clipped)

        pre_vol = pre_q.loc[i, 'volatility_sd']
        post_vol = post_q.loc[i, 'volatility_sd']
        pct_vol = (post_vol - pre_vol) / pre_vol if pre_vol > 0 else 0
        beta_vol = pct_vol / STT_CHANGE
        beta_vol_list.append(beta_vol)

    results = {
        'BETA_TV_MIN': min(beta_tv_list) if beta_tv_list else -0.5,
        'BETA_TV_MAX': max(beta_tv_list) if beta_tv_list else 0.5,
        'BETA_TV_MEDIAN': float(np.median(beta_tv_list)) if beta_tv_list else 0.0,
        'BETA_VOL_MIN': min(beta_vol_list) if beta_vol_list else -0.5,
        'BETA_VOL_MAX': max(beta_vol_list) if beta_vol_list else 0.5,
        'BETA_VOL_MEDIAN': float(np.median(beta_vol_list)) if beta_vol_list else 0.0,
    }

    print(f"\n    Volume Elasticity (β_TV):   [{results['BETA_TV_MIN']:>+8.4f}, {results['BETA_TV_MAX']:>+8.4f}]")
    print(f"    Volatility Elasticity (β_VOL): [{results['BETA_VOL_MIN']:>+8.4f}, {results['BETA_VOL_MAX']:>+8.4f}]")

    render_table(
        title="Table 2.  Elasticity Ranges (TRAIN Law Natural Experiment, 2018)",
        col_labels=['Elasticity', 'Min', 'Median', 'Max'],
        row_data=[
            ['Volume Elasticity (β_TV)', f'{results["BETA_TV_MIN"]:.4f}',
             f'{results["BETA_TV_MEDIAN"]:.4f}', f'{results["BETA_TV_MAX"]:.4f}'],
            ['Volatility Elasticity (β_VOL)', f'{results["BETA_VOL_MIN"]:.4f}',
             f'{results["BETA_VOL_MEDIAN"]:.4f}', f'{results["BETA_VOL_MAX"]:.4f}'],
        ],
        fname='table2_elasticity.png',
        col_widths=[0.28, 0.24, 0.24, 0.24],
        figsize=(14, 2.8),
        footnote=(f"Calculated from {n_q} quarterly observations (Jul 2016 – Jun 2019)."),
    )

    return results


# ============================================================================
# STAGE 3 — LAMBDA FACTORS
# ============================================================================

def calculate_lambda_ranges(pse_data: dict, med_data: dict) -> dict:
    """STAGE 3: Lambda mediating factors."""
    _section_header(3, "LAMBDA MEDIATING FACTOR DERIVATION")

    flr = med_data['flr'].copy()
    pre_flr = flr[flr['Period'] == 'Pre-CMEPA'].dropna(subset=['FLR']).copy()

    if len(pre_flr) > 0:
        flr_mean = pre_flr['FLR'].mean()
        pre_flr['lambda'] = pre_flr['FLR'] / flr_mean
        lam_mean = float(pre_flr['lambda'].mean())
        lam_std = float(pre_flr['lambda'].std(ddof=1)) if len(pre_flr) > 1 else 0.1
    else:
        lam_mean = 1.0
        lam_std = 0.15

    print(f"\n    λ_mean = {lam_mean:.4f}")
    print(f"    λ_std  = {lam_std:.4f}")

    render_table(
        title="Table 3.  Lambda Mediating Factors (Pre-CMEPA, Jul 2024 – Jun 2025)",
        col_labels=['Parameter', 'Mean', 'Std Dev', 'Definition'],
        row_data=[
            ['Composite Lambda (λ)', f'{lam_mean:.4f}', f'{lam_std:.4f}',
             'Market condition scaling factor'],
        ],
        fname='table3_lambda_parameters.png',
        col_widths=[0.30, 0.20, 0.20, 0.30],
        figsize=(14, 2.5),
        footnote="λ amplifies (>1) or dampens (<1) the elasticity effect based on market sentiment.",
    )

    return {'lam_mean': lam_mean, 'lam_std': lam_std}


# ============================================================================
# STAGE 4 — MONTE CARLO SIMULATION
# ============================================================================

def _run_single_horizon(label: str, cfg: dict, eda_params: dict,
                        elasticity: dict, lambda_ranges: dict) -> dict:
    """STAGE 4: Runs one time-horizon simulation."""

    sims_per_day = cfg['sims_per_day']
    trading_days = cfg['trading_days']
    total_iters = cfg['total_iters']

    TV_MEAN = eda_params['TV_mu']
    TV_SD = eda_params['TV_sigma']
    P_AVG = eda_params['P_avg']
    VOL_PRE = eda_params['VOL_pre']

    beta_tv_range = (elasticity['BETA_TV_MIN'], elasticity['BETA_TV_MAX'])
    beta_vol_range = (elasticity['BETA_VOL_MIN'], elasticity['BETA_VOL_MAX'])
    lam_mean, lam_std = lambda_ranges['lam_mean'], lambda_ranges['lam_std']

    DAYS_BASELINE = 126
    REV_BASELINE = (TV_MEAN * P_AVG) * STT_PRE * DAYS_BASELINE

    _subsection(f"{label} HORIZON MONTE CARLO SIMULATION")
    print(f"    Total iterations: {total_iters:,}")

    daily_tv_pct_mean = np.zeros(trading_days)
    daily_vol_pct_mean = np.zeros(trading_days)
    daily_rev_mean = np.zeros(trading_days)
    daily_rev_std = np.zeros(trading_days)
    daily_tv_post_mean = np.zeros(trading_days)

    _n_flat = min(total_iters, 50_000)
    flat_beta_tv = np.zeros(_n_flat)
    flat_beta_vol = np.zeros(_n_flat)
    flat_lam = np.zeros(_n_flat)
    flat_tv_pre = np.zeros(_n_flat)
    flat_rev = np.zeros(_n_flat)
    _flat_idx = 0

    print(f"    Running {total_iters:,} iterations...")
    print(f"    {'─' * 76}")

    for d in range(trading_days):

        if (d + 1) % max(1, trading_days // 10) == 0:
            progress = (d + 1) / trading_days * 100
            print(f"    Progress: {progress:>5.1f}% complete ({d+1:>5,} / {trading_days:,} days)")

        tv_pre_arr = np.random.normal(TV_MEAN, TV_SD, sims_per_day)
        tv_pre_arr = np.maximum(tv_pre_arr, TV_MEAN * 0.05)

        beta_tv_arr = np.random.uniform(*beta_tv_range, sims_per_day)
        beta_vol_arr = np.random.uniform(*beta_vol_range, sims_per_day)
        lam_arr = np.random.normal(lam_mean, lam_std, sims_per_day)
        lam_arr = np.clip(lam_arr, 0.5, 2.0)

        pct_tv_arr = beta_tv_arr * STT_CHANGE_PCT * lam_arr
        pct_vol_arr = beta_vol_arr * STT_CHANGE_PCT * lam_arr

        tv_post_arr = tv_pre_arr * (1 + pct_tv_arr)
        gsp_post_arr = tv_post_arr * P_AVG
        rev_post_arr = gsp_post_arr * STT_POST
        vol_post_arr = VOL_PRE * (1 + pct_vol_arr)

        daily_tv_pct_mean[d] = pct_tv_arr.mean()
        daily_vol_pct_mean[d] = pct_vol_arr.mean()
        daily_rev_mean[d] = rev_post_arr.mean()
        daily_rev_std[d] = rev_post_arr.std()
        daily_tv_post_mean[d] = tv_post_arr.mean()

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

    print(f"    {'─' * 76}")

    cumulative_rev = float(daily_rev_mean.sum())
    mean_tv_pct = float(daily_tv_pct_mean.mean()) * 100
    mean_vol_pct = float(daily_vol_pct_mean.mean()) * 100
    pct_vol_increase = float((daily_vol_pct_mean > 0).mean()) * 100

    rev_counterfactual = float((daily_tv_post_mean * P_AVG * STT_PRE).sum())
    laffer_ok = cumulative_rev > rev_counterfactual
    rev_chg_pct = (cumulative_rev - rev_counterfactual) / max(rev_counterfactual, 1) * 100

    ci_low = float((daily_rev_mean - 1.96 * daily_rev_std).sum())
    ci_high = float((daily_rev_mean + 1.96 * daily_rev_std).sum())

    print(f"    Daily mean % Δ Volume:       {mean_tv_pct:>+10.4f}%")
    print(f"    Daily mean % Δ Volatility:   {mean_vol_pct:>+10.4f}%")
    print(f"    Cumulative Revenue (Rev_S):  ₱{cumulative_rev:>15,.2f}")
    print(f"    Laffer Verdict:              {'ADEQUATE ✓' if laffer_ok else 'INADEQUATE ✗':>12}")

    return {
        'label': label,
        'horizon_years': cfg['horizon_years'],
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
    """Runs all three horizons."""
    _section_header(4, "MULTI-HORIZON MONTE CARLO SIMULATION")

    horizon_results = {}
    for label, cfg in HORIZONS.items():
        horizon_results[label] = _run_single_horizon(
            label, cfg, eda_params, elasticity, lambda_ranges)

    print(f"\n\n    ══ CONSOLIDATED RESULTS ACROSS ALL HORIZONS ══")
    print(f"    {'Metric':<50} {'1-Year':>14}  {'5-Year':>14}  {'10-Year':>14}")
    print("    " + "─" * 95)

    metrics_rows = []
    for metric_lbl, key, fmt in [
        ('Total Iterations', 'total_iters', lambda v: f'{v:,}'),
        ('Daily mean % Δ Volume', 'mean_tv_pct', lambda v: f'{v:+.4f}%'),
        ('Daily mean % Δ Volatility', 'mean_vol_pct', lambda v: f'{v:+.4f}%'),
        ('Cumulative Revenue (Rev_S)', 'cumulative_rev', lambda v: f'₱{v/1e6:,.2f}M'),
        ('Laffer Verdict', 'laffer_ok', lambda v: 'ADEQUATE ✓' if v else 'INADEQUATE ✗'),
    ]:
        vals = [horizon_results[h][key] for h in HORIZONS]
        row = [metric_lbl] + [fmt(v) for v in vals]
        metrics_rows.append(row)
        print(f"    {metric_lbl:<50} {fmt(vals[0]):>14}  {fmt(vals[1]):>14}  {fmt(vals[2]):>14}")

    render_table(
        title="Table 4.  Consolidated Monte Carlo Results — All Time Horizons",
        col_labels=['Metric', '1-Year (252K)', '5-Year (630K)', '10-Year (630K)'],
        row_data=metrics_rows,
        fname='table4_horizon_results.png',
        col_widths=[0.42, 0.19, 0.19, 0.20],
        figsize=(16, 4),
        highlight_rows=[0, 1, 4],
        footnote=("Daily % changes identical across horizons (stationarity).\n"
                  "Revenue scales linearly with trading days."),
    )

    return horizon_results


# ============================================================================
# STAGE 5 — SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(horizon_results: dict) -> dict:
    """Sensitivity analysis."""
    _section_header(5, "SENSITIVITY ANALYSIS (Spearman Rank Correlation)")

    all_sa = {}
    all_sa_rows = []

    for label, res in horizon_results.items():
        inputs = {
            'β_TV': res['flat_beta_tv'],
            'β_VOL': res['flat_beta_vol'],
            'TV_pre': res['flat_tv_pre'],
            'λ': res['flat_lam'],
        }
        output = res['flat_rev']

        rows = []
        for name, data in inputs.items():
            rho, pval = spearmanr(data, output)
            rows.append({'Variable': name, 'Rho': rho, 'AbsRho': abs(rho), 'PValue': pval})

        df_sa = pd.DataFrame(rows).sort_values('AbsRho', ascending=True).reset_index(drop=True)
        all_sa[label] = df_sa

        t_rows = [[label, str(n+1), row['Variable'], f'{row["Rho"]:+.4f}']
                  for n, (_, row) in enumerate(df_sa.iterrows())]
        all_sa_rows.extend(t_rows)

    render_table(
        title="Table 5.  Sensitivity Analysis — Spearman Rank Correlations",
        col_labels=['Horizon', 'Rank', 'Input', 'ρ'],
        row_data=all_sa_rows,
        fname='table5_sensitivity_all_horizons.png',
        col_widths=[0.15, 0.10, 0.50, 0.25],
        figsize=(14, 4),
        footnote=("Spearman ρ: rank correlation, robust to outliers."),
    )

    return all_sa


# ============================================================================
# STAGE 6 — MODEL VALIDATION TEST (COMPLETE)
# ============================================================================

def run_model_validation(horizon_results: dict, post_cmepa: pd.DataFrame,
                         eda_params: dict) -> dict:
    """
    STAGE 6: COMPLETE MODEL VALIDATION TEST

    Three-part validation:
    1. 95% CI Test (Prediction bounds vs actual)
    2. Distribution Normality Test (K-S, Shapiro-Wilk)
    3. Residual Analysis (Predicted vs Actual)
    """

    _section_header(6, "MODEL VALIDATION TEST (COMPLETE ASSESSMENT)")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: 95% CONFIDENCE INTERVAL TEST
    # ────────────────────────────────────��────────────────────────────────────
    _subsection("PART 1: 95% CONFIDENCE INTERVAL TEST")

    print("\n    1A. EXTRACT ACTUAL POST-CMEPA DATA")
    print(f"    {'─' * 76}")

    if len(post_cmepa) > 0:
        actual_volume = float(post_cmepa['Volume'].mean()) if 'Volume' in post_cmepa.columns else 0
        actual_price = float(post_cmepa['Closing Price'].mean()) if 'Closing Price' in post_cmepa.columns else 1
    else:
        actual_volume, actual_price = 0, 1

    actual_gsp = actual_volume * actual_price
    actual_rev = actual_gsp * STT_POST * len(post_cmepa) if len(post_cmepa) > 0 else 0
    actual_vol_obs = float(post_cmepa['Volatility'].mean()) if 'Volatility' in post_cmepa.columns and len(post_cmepa) > 0 else 0

    pre_volume_mean = eda_params['pre_tv_mean']
    actual_volume_chg_pct = ((actual_volume - pre_volume_mean) / pre_volume_mean * 100) if pre_volume_mean > 0 else 0

    print(f"    Actual Post-CMEPA Observations (Jul–Dec 2025, n={len(post_cmepa)}):")
    print(f"      Mean Daily Volume:           {actual_volume:>20,.0f} shares")
    print(f"      vs Pre-CMEPA Mean:           {pre_volume_mean:>20,.0f} shares")
    print(f"      Observed Volume Change:      {actual_volume_chg_pct:>+19.2f}%")
    print(f"      Actual Daily GSP:            ₱{actual_gsp:>19,.2f}")
    print(f"      Actual Total Revenue (6m):   ₱{actual_rev:>19,.2f}")
    print(f"      Observed Market Volatility:  {actual_vol_obs:>20.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1B: 1-YEAR HORIZON VALIDATION (Primary)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n    1B. 1-YEAR HORIZON PREDICTION BOUNDS (95% CI)")
    print(f"    {'─' * 76}")

    res_1y = horizon_results['1-Year']

    # Volume % change CI
    tv_pct_daily = res_1y['daily_tv_pct_mean'] * 100
    tv_pct_mean = float(tv_pct_daily.mean())
    tv_pct_std = float(tv_pct_daily.std())
    ci_tv_pct_lo = tv_pct_mean - 1.96 * tv_pct_std
    ci_tv_pct_hi = tv_pct_mean + 1.96 * tv_pct_std
    tv_pct_in_ci = bool(ci_tv_pct_lo <= actual_volume_chg_pct <= ci_tv_pct_hi)

    # Revenue CI
    ci_rev_lo = res_1y['ci_low']
    ci_rev_hi = res_1y['ci_high']
    rv_in_ci = bool(ci_rev_lo <= actual_rev <= ci_rev_hi) if actual_rev > 0 else None

    # Volatility CI
    vol_daily = res_1y['daily_vol_pct_mean']
    vol_mean = float(vol_daily.mean()) * 100
    vol_std = float(vol_daily.std()) * 100
    ci_vol_lo = vol_mean - 1.96 * vol_std
    ci_vol_hi = vol_mean + 1.96 * vol_std

    print(f"    Metric                      Predicted (±95% CI)          Actual      In CI?")
    print(f"    {'─' * 76}")
    print(f"    % Δ Volume (daily mean)     {tv_pct_mean:>+8.4f}% [{ci_tv_pct_lo:>+8.4f}%, {ci_tv_pct_hi:>+8.4f}%]  {actual_volume_chg_pct:>+8.2f}%   "
          f"{'✓ YES' if tv_pct_in_ci else '✗ NO'}")
    print(f"    Total Revenue (₱)           ₱{res_1y['cumulative_rev']/1e6:>8.2f}M [₱{ci_rev_lo/1e6:>8.2f}M, ₱{ci_rev_hi/1e6:>8.2f}M]  "
          f"₱{actual_rev/1e6:>8.2f}M  {'✓ YES' if rv_in_ci else '✗ NO' if rv_in_ci is not None else 'N/A'}")
    print(f"    % Δ Volatility (daily mean) {vol_mean:>+8.4f}% [{ci_vol_lo:>+8.4f}%, {ci_vol_hi:>+8.4f}%]  N/A      N/A")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: DISTRIBUTION NORMALITY TESTS
    # ─────────────────────────────────────────────────────────────────────────
    _subsection("PART 2: DISTRIBUTION NORMALITY TESTS")

    print(f"\n    2A. KOLMOGOROV-SMIRNOV TEST (vs Normal Distribution)")
    print(f"    {'─' * 76}")

    # Normalize the daily revenues
    rev_normalized = (res_1y['daily_rev_mean'] - res_1y['daily_rev_mean'].mean()) / res_1y['daily_rev_mean'].std()
    ks_stat, ks_pval = kstest(rev_normalized, 'norm')

    print(f"    Null Hypothesis: Daily revenues follow Normal distribution")
    print(f"    {'─' * 76}")
    print(f"    KS Test Statistic:           {ks_stat:>20.6f}")
    print(f"    p-value:                     {ks_pval:>20.6f}")
    print(f"    Significance Level (α):      {0.05:>20.4f}")
    print(f"    Result:                      {'FAIL TO REJECT H0 ✓' if ks_pval > 0.05 else 'REJECT H0 ✗':>20}")
    print(f"    Interpretation:              {'Data consistent with Normal' if ks_pval > 0.05 else 'Data deviates from Normal':>20}")

    print(f"\n    2B. SHAPIRO-WILK TEST (Sample Normality)")
    print(f"    {'─' * 76}")

    # Sample for Shapiro-Wilk (max 5000 samples)
    sample_size = min(5000, len(res_1y['daily_rev_mean']))
    sample_indices = np.random.choice(len(res_1y['daily_rev_mean']), sample_size, replace=False)
    sample_data = res_1y['daily_rev_mean'][sample_indices]

    try:
        from scipy.stats import shapiro
        sw_stat, sw_pval = shapiro(sample_data)
        print(f"    Sample Size:                 {sample_size:>20}")
        print(f"    Shapiro-Wilk Statistic:      {sw_stat:>20.6f}")
        print(f"    p-value:                     {sw_pval:>20.6f}")
        print(f"    Result:                      {'NORMAL ✓' if sw_pval > 0.05 else 'NON-NORMAL ✗':>20}")
    except:
        print(f"    Result:                      Shapiro-Wilk unavailable")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: RESIDUAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    _subsection("PART 3: RESIDUAL ANALYSIS (Prediction Accuracy)")

    print(f"\n    3A. ERROR METRICS")
    print(f"    {'─' * 76}")

    # Revenue residual
    predicted_revenue = res_1y['cumulative_rev']
    revenue_error = actual_rev - predicted_revenue
    revenue_error_pct = (revenue_error / max(predicted_revenue, 1)) * 100
    revenue_mape = abs(revenue_error_pct)

    # Volume residual
    predicted_volume_pct = res_1y['mean_tv_pct']
    volume_error_pct = actual_volume_chg_pct - predicted_volume_pct

    print(f"    Revenue Prediction Error:")
    print(f"      Predicted Revenue (Rev_S):  ₱{predicted_revenue:>18,.2f}")
    print(f"      Actual Revenue (observed):  ₱{actual_rev:>18,.2f}")
    print(f"      Absolute Error (₱):         ₱{abs(revenue_error):>18,.2f}")
    print(f"      % Error:                    {revenue_error_pct:>+18.2f}%")
    print(f"      MAPE (Mean Absolute % Error): {revenue_mape:>14.2f}%")

    print(f"\n    Volume % Change Prediction Error:")
    print(f"      Predicted % Δ Volume:       {predicted_volume_pct:>+18.4f}%")
    print(f"      Actual % Δ Volume:          {actual_volume_chg_pct:>+18.4f}%")
    print(f"      Error:                      {volume_error_pct:>+18.4f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # OVERALL ROBUSTNESS VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    _subsection("PART 4: OVERALL ROBUSTNESS VERDICT")

    print(f"\n    VALIDATION CHECKLIST:")
    print(f"    {'─' * 76}")

    checks = [
        ("Volume within 95% CI", tv_pct_in_ci),
        ("Revenue within 95% CI", rv_in_ci if rv_in_ci is not None else False),
        ("KS Test: Normal Distribution (p > 0.05)", ks_pval > 0.05),
        ("Revenue MAPE < 15%", revenue_mape < 15),
    ]

    passes = sum([1 for _, result in checks if result])
    total_checks = len(checks)

    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"    {check_name:<45} {status:>20}")

    print(f"\n    {'─' * 76}")
    print(f"    Checks Passed: {passes}/{total_checks}")

    if passes == total_checks:
        robustness = "ROBUST ✓✓"
        robustness_color = C_GREEN
    elif passes >= total_checks - 1:
        robustness = "MOSTLY ROBUST ✓"
        robustness_color = C_GREEN
    else:
        robustness = "WEAK ✗"
        robustness_color = C_RED

    print(f"    Overall Assessment: {robustness}")
    print(f"    {'─' * 76}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    validation_rows = [
        ['95% CI: Volume % Δ', f'{ci_tv_pct_lo:+.4f}% – {ci_tv_pct_hi:+.4f}%',
         f'{actual_volume_chg_pct:+.2f}%', '✓ YES' if tv_pct_in_ci else '✗ NO'],
        ['95% CI: Total Revenue', f'₱{ci_rev_lo/1e6:,.2f}M – ₱{ci_rev_hi/1e6:,.2f}M',
         f'₱{actual_rev/1e6:,.2f}M', '✓ YES' if rv_in_ci else ('✗ NO' if rv_in_ci is not None else 'N/A')],
        ['Kolmogorov-Smirnov Test', f'p = {ks_pval:.4f}', 'α = 0.05', '✓ PASS' if ks_pval > 0.05 else '✗ FAIL'],
        ['Revenue MAPE', f'{revenue_mape:.2f}%', '< 15%', '✓ PASS' if revenue_mape < 15 else '✗ FAIL'],
        ['Overall Robustness', '—', '—', robustness],
    ]

    render_table(
        title="Table 7.  Complete Model Validation Test Results\n"
              "95% CI, Normality Tests, Residual Analysis, and Robustness Assessment",
        col_labels=['Test', 'Predicted/Test Value', 'Actual/Threshold', 'Result'],
        row_data=validation_rows,
        fname='table7_model_validation_complete.png',
        col_widths=[0.25, 0.30, 0.20, 0.25],
        figsize=(16, 4),
        highlight_rows=[4],
        footnote=("Complete validation includes 95% CI bounds (volume, revenue), "
                  "normality tests (KS, Shapiro-Wilk), and residual analysis (MAPE).\n"
                  "Model is ROBUST if ≥4/4 checks pass."),
    )

    return {
        'tv_pct_in_ci': tv_pct_in_ci,
        'rv_in_ci': rv_in_ci,
        'ks_pval': ks_pval,
        'ks_stat': ks_stat,
        'revenue_mape': revenue_mape,
        'revenue_error_pct': revenue_error_pct,
        'robustness': robustness,
        'checks_passed': passes,
        'total_checks': total_checks,
    }


# ============================================================================
# STAGE 7 — FINAL SUMMARY
# ============================================================================

def print_final_summary(horizon_results: dict, validation: dict) -> None:
    """Final summary."""
    _section_header(7, "FINAL RESEARCH SUMMARY")

    summary_rows = [
        ['1-YEAR HORIZON RESULTS', ''],
        ['  Total Iterations', f'{horizon_results["1-Year"]["total_iters"]:,}'],
        ['  Daily mean % Δ Volume', f'{horizon_results["1-Year"]["mean_tv_pct"]:+.4f}%'],
        ['  Cumulative Revenue (Rev_S)', f'₱{horizon_results["1-Year"]["cumulative_rev"]/1e6:,.2f}M'],
        [''],
        ['LAFFER FISCAL ADEQUACY', ''],
        ['  1-Year Horizon', '✓ ADEQUATE' if horizon_results['1-Year']['laffer_ok'] else '✗ INADEQUATE'],
        ['  5-Year Horizon', '✓ ADEQUATE' if horizon_results['5-Year']['laffer_ok'] else '✗ INADEQUATE'],
        ['  10-Year Horizon', '✓ ADEQUATE' if horizon_results['10-Year']['laffer_ok'] else '✗ INADEQUATE'],
        [''],
        ['MODEL VALIDATION RESULTS', ''],
        [f'  Validation Checks Passed', f'{validation["checks_passed"]}/{validation["total_checks"]}'],
        ['  Overall Robustness', validation['robustness']],
        ['  Volume 95% CI Test', '✓ PASS' if validation['tv_pct_in_ci'] else '✗ FAIL'],
        ['  Revenue 95% CI Test', '✓ PASS' if validation['rv_in_ci'] else ('✗ FAIL' if validation['rv_in_ci'] is not None else 'N/A')'],
        ['  KS Normality Test (p > 0.05)', '✓ PASS' if validation['ks_pval'] > 0.05 else '✗ FAIL'],
        ['  Revenue MAPE', f'{validation["revenue_mape"]:.2f}% (< 15% ✓)' if validation["revenue_mape"] < 15 else f'{validation["revenue_mape"]:.2f}% (≥ 15% ✗)'],
    ]

    render_table(
        title="Table 8.  Final Research Summary & Validation Results\n"
              "Monte Carlo Simulation Results + Complete Model Validation Assessment",
        col_labels=['Finding / Metric', 'Value'],
        row_data=summary_rows,
        fname='table8_final_summary.png',
        col_widths=[0.55, 0.45],
        figsize=(15, 9),
        highlight_rows=[0, 5, 10],
        footnote=("Validation includes 95% CI test, normality tests (K-S), and residual analysis (MAPE).\n"
                  "Model is ROBUST if all validation checks pass."),
    )

    print("\n" + "="*80)
    print("  SIMULATION COMPLETE")
    print("="*80)
    print(f"\n  Output Directory: {OUTPUT_DIR}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  FILES GENERATED:")
    print(f"    - table7_model_validation_complete.png  ← VALIDATION TEST")
    print(f"    - table8_final_summary.png")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    print("="*80)
    print("  CMEPA MONTE CARLO SIMULATION — COMPLETE WITH MODEL VALIDATION")
    print("="*80)

    log_message("Starting simulation pipeline", "MAIN")

    pse_data = load_pse_data(PSE_FILE)
    med_data = load_mediating_data(MED_FILE)

    eda_params = calculate_eda_parameters(pse_data)
    elasticity = estimate_elasticity(pse_data['pre_train'], pse_data['post_train'])
    lambda_ranges = calculate_lambda_ranges(pse_data, med_data)

    horizon_results = run_all_horizons(eda_params, elasticity, lambda_ranges)
    sensitivity = run_sensitivity_analysis(horizon_results)

    # ✅ COMPLETE MODEL VALIDATION TEST
    validation = run_model_validation(horizon_results, pse_data['post_cmepa'], eda_params)

    print_final_summary(horizon_results, validation)

    log_message("Simulation pipeline complete", "MAIN")
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")