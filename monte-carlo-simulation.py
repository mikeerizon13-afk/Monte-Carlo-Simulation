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

─── PIPELINE OVERVIEW ───────────────────────────────────────────────────────
  Stage 0 │ Data Loading & Cleaning
  Stage 1 │ Estimated Distribution Parameter Calculation (Jul–Dec 2024 baseline)
  Stage 2 │ Elasticity Parameter Estimation (TRAIN Law natural experiment)
  Stage 3 │ Lambda (λ) Mediating Factor Derivation [EDA Normal Sampling]
  Stage 4 │ Time-Horizon Monte Carlo Simulation (Full Iteration Storage)
             1-Year  : 1,000 sims/day × 252 trading days  =  252,000 iterations
             5-Year  : 500  sims/day × 1,260 trading days =  630,000 iterations
             10-Year : 250  sims/day × 2,520 trading days =  630,000 iterations
  Stage 5 │ Export Results to Excel
  Stage 6 │ Sensitivity Analysis (Spearman's ρ, Tornado Chart)
  Stage 7 │ Monte Carlo Distribution Visualisations
  Stage 8 │ Policy Shock Transmission Flowchart
  Stage 9 │ Final Research Summary

─── KEY DESIGN DECISIONS ────────────────────────────────────────────────────
  Revenue unit    : flat_rev stores DAILY single-day revenue (₱/day).
                    6-month projected revenue = mean_rev_per_day × 126 days.
                    REV_BASELINE uses the same 126-day window for comparability.
  Laffer test     : cumulative_rev (Rev_S) vs. REV_BASELINE (Rev_B) per horizon.
  Lambda derivation: ALL three λ components are derived from empirical pre-CMEPA
                     data — no hardcoded placeholders.
=============================================================================
"""

import pandas              as pd
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.ticker   as mticker
from   matplotlib.patches  import FancyBboxPatch
from   matplotlib.gridspec import GridSpec
from   scipy.stats         import spearmanr, norm as sp_norm
import warnings
warnings.filterwarnings('ignore')

# ─── GLOBAL STYLE ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'axes.edgecolor':   '#333333',
    'axes.linewidth':   0.8,
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'text.color':       '#1a1a1a',
    'axes.labelcolor':  '#1a1a1a',
    'grid.color':       '#dddddd',
    'grid.linewidth':   0.5,
})

# ─── GLOBAL CONSTANTS ────────────────────────────────────────────────────────
STT_PRE        = 0.006                           # 0.6 % (TRAIN Law / Pre-CMEPA)
STT_POST       = 0.001                           # 0.1 % (CMEPA)
STT_CHANGE_PCT = (STT_POST - STT_PRE) / STT_PRE  # −83.33 %
BETA_TV_CLIP   = 2.5                             # outlier cap for β_TV
DAYS_6M        = 126                             # approximate 6-month trading days
TRAIN_STT_CHANGE = 0.20                          # +20 % STT change under TRAIN Law

# Time-horizon structure
HORIZONS = {
    '1-Year':  {'sims_per_day': 1_000, 'trading_days':   252},
    '5-Year':  {'sims_per_day':   500, 'trading_days': 1_260},
    '10-Year': {'sims_per_day':   250, 'trading_days': 2_520},
}

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
C_NAVY  = '#1a2e4a'
C_BLUE  = '#1a4f8a'
C_RED   = '#8b1a1a'
C_GREEN = '#1a5c2e'
C_AMBER = '#b8860b'
C_DARK  = '#1a1a1a'
C_LIGHT = '#f5f5f5'

# ─── FILE PATHS ──────────────────────────────────────────────────────────────
# Update these to match the exact filenames in your working directory.
# Common variations depending on how the file was saved / uploaded:
#   "PSE Dataset (6).xlsx"                 ← spaces + parentheses
#   "PSE_Dataset__6_.xlsx"                 ← underscores
#   "PSE_Dataset_(6).xlsx"                 ← underscores + parentheses
import os as _os

def _find_file(candidates: list) -> str:
    """Returns the first filename from the candidate list that exists on disk.
    Raises a clear FileNotFoundError listing all tried paths if none is found."""
    for name in candidates:
        if _os.path.isfile(name):
            return name
    tried = "\n    ".join(candidates)
    raise FileNotFoundError(
        f"Could not find the data file. Tried:\n    {tried}\n"
        f"Please rename your file to one of the above or update the "
        f"PSE_FILE / MED_FILE constants at the top of this script.")

PSE_FILE = _find_file([
    "PSE Dataset (1).xlsx",

])

MED_FILE = _find_file([
    "Mediating Variables - Dataset (1).xlsx",
])

print(f"[ Config ] PSE file  : {PSE_FILE}")
print(f"[ Config ] MED file  : {MED_FILE}")


# =============================================================================
# UTILITY — PROFESSIONAL TABLE RENDERER
# =============================================================================

def render_table(title: str, col_labels: list, row_data: list, fname: str = None,
                 col_widths: list = None, footnote: str = None,
                 highlight_rows: list = None, figsize: tuple = None) -> plt.Figure:
    """Renders publication-quality tables (white background, Times New Roman)."""
    n_cols = len(col_labels)
    n_rows = len(row_data)

    if figsize is None:
        figsize = (max(9, 1.5 * n_cols), max(2.0, 0.44 * n_rows + 1.6))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    cell_colours = [['white'] * n_cols for _ in range(n_rows)]
    bbox_y0 = 0.07 if footnote else 0.03
    tbl = ax.table(
        cellText=row_data, colLabels=col_labels, cellLoc='center', loc='center',
        cellColours=cell_colours, bbox=[0.01, bbox_y0, 0.98, 1.0 - bbox_y0 - 0.05])
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
    else:
        tbl.auto_set_column_width(range(n_cols))

    fig.suptitle(title, fontsize=10, fontweight='bold', x=0.5, y=0.99,
                 ha='center', va='top', color=C_DARK, fontfamily='serif')
    if footnote:
        fig.text(0.015, 0.005, footnote, fontsize=7.5, color='#444444',
                 style='italic', va='bottom', fontfamily='serif')

    plt.tight_layout(rect=[0, 0.05 if footnote else 0, 1, 0.96])
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"    [Table PNG saved → {fname}]")
    plt.show()
    return fig


def _section_header(stage: int, title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  STAGE {stage} — {title}\n{bar}")


def _subsection(title: str) -> None:
    print(f"\n  {'─'*74}\n    {title}\n  {'─'*74}")


# =============================================================================
# STAGE 0 — DATA LOADING & CLEANING
# =============================================================================

def load_pse_data(file: str) -> dict:
    """
    Loads the four PSE study-period DataFrames.

    Sheet mapping:
      'Trading Volume Pre-TRAIN'   → Jul 2016 – Dec 2017  (STT 0.5 %)
      'Trading Volume Post-TRAIN'  → Jan 2018 – Jun 2019  (STT 0.6 %)
      'Trading Volume Pre-CMEPA'   → Jul 2024 – Jun 2025  (STT 0.6 %)
      'Trading Volume Post-CMEPA'  → Jul 2025 – Dec 2025  (STT 0.1 %)
    """
    def _clean(sheet: str, date_lo: str = None, date_hi: str = None) -> pd.DataFrame:
        df = pd.read_excel(file, sheet_name=sheet)
        df['Date']          = pd.to_datetime(df['Date'],         errors='coerce')
        df['Closing Price'] = pd.to_numeric(df['Closing Price'], errors='coerce')
        df['Trade Value']   = pd.to_numeric(df['Trade Value'],   errors='coerce')
        df['Daily Returns'] = pd.to_numeric(df['Daily Returns'], errors='coerce')
        df['Volume']        = pd.to_numeric(
            df.get('Volume', pd.Series(dtype=float)), errors='coerce')
        df['Volatility']    = pd.to_numeric(
            df['SHORT TERM VOLATILITY (5-DAY WINDOW)'], errors='coerce')
        df = df.dropna(subset=['Date', 'Trade Value', 'Closing Price'])
        df = df.sort_values('Date').reset_index(drop=True)
        if date_lo:
            df = df[df['Date'] >= date_lo]
        if date_hi:
            df = df[df['Date'] <= date_hi]
        return df.reset_index(drop=True)

    pre_train  = _clean('Trading Volume Pre-TRAIN',  '2016-07-01', '2017-12-31')
    post_train = _clean('Trading Volume Post-TRAIN', '2018-01-01', '2019-06-30')
    pre_cmepa  = _clean('Trading Volume Pre-CMEPA',  '2024-07-01', '2025-06-30')
    post_cmepa = _clean('Trading Volume Post-CMEPA', '2025-07-01')

    print("[ Stage 0 ] PSE Data Loaded")
    for lbl, df in [('Pre-TRAIN', pre_train), ('Post-TRAIN', post_train),
                    ('Pre-CMEPA', pre_cmepa), ('Post-CMEPA', post_cmepa)]:
        if len(df):
            print(f"    {lbl:<14}: {len(df):>4} trading days  "
                  f"({df['Date'].min().date()} – {df['Date'].max().date()})")
        else:
            print(f"    {lbl:<14}: 0 rows — check sheet / date filters")

    return {'pre_train': pre_train, 'post_train': post_train,
            'pre_cmepa': pre_cmepa, 'post_cmepa': post_cmepa}


def load_mediating_data(file: str) -> dict:
    """
    Loads all mediating-variable datasets.

    Sheets: GDPGrowth_Data, InflationRate_Data, InvestorType_Data (FLR).
    All three datasets are consumed by calculate_lambda_ranges (Stage 3).
    """
    # GDP Growth
    gdp = pd.read_excel(file, sheet_name='GDPGrowth_Data', header=3)
    gdp.columns = ['Year', 'Quarter', 'Period', 'GDP_Growth', 'Months', 'Note']
    gdp['Year']       = pd.to_numeric(gdp['Year'],       errors='coerce')
    gdp['GDP_Growth'] = pd.to_numeric(gdp['GDP_Growth'], errors='coerce')
    gdp = gdp.dropna(subset=['Year', 'GDP_Growth']).reset_index(drop=True)

    # Inflation Rate
    inf = pd.read_excel(file, sheet_name='InflationRate_Data', header=3)
    inf.columns = ['Year', 'Month', 'InflationRate', 'Classification', 'Note']
    inf['Year']          = pd.to_numeric(inf['Year'],          errors='coerce')
    inf['InflationRate'] = pd.to_numeric(inf['InflationRate'], errors='coerce')
    inf = inf.dropna(subset=['Year', 'InflationRate']).reset_index(drop=True)
    _month_map = {m: i + 1 for i, m in enumerate([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'])}
    inf['MonthNum'] = inf['Month'].map(_month_map)

    # Investor Type (Foreign-Local Ratio)
    flr = pd.read_excel(file, sheet_name='InvestorType_Data', header=2)
    flr.columns = ['Period', 'Month', 'TotalValue', 'AvgDaily',
                   'ForeignPct', 'ForeignValue', 'LocalPct', 'FLR']
    for col in ['TotalValue', 'ForeignPct', 'ForeignValue', 'LocalPct', 'FLR']:
        flr[col] = pd.to_numeric(flr[col], errors='coerce')
    flr = flr.dropna(subset=['FLR']).reset_index(drop=True)

    print("[ Stage 0 ] Mediating Variable Data Loaded")
    print(f"    GDP rows      : {len(gdp)}")
    print(f"    Inflation rows: {len(inf)}")
    print(f"    Investor (FLR): {len(flr)} months")

    return {'gdp': gdp, 'inflation': inf, 'flr': flr}


# =============================================================================
# STAGE 1 — ESTIMATED DISTRIBUTION PARAMETER CALCULATION
# =============================================================================

def calculate_eda_parameters(pse_data: dict) -> dict:
    """
    Computes EDA baseline parameters from the Jul–Dec 2024 window.

    This 6-month window is seasonally aligned with the post-CMEPA evaluation
    period (Jul–Dec 2025), eliminating seasonal bias from the Laffer comparison.

    Returns
    -------
    TV_mu        : mean daily share volume (Normal dist. location)
    TV_sigma     : std dev daily share volume (Normal dist. scale)
    P_avg        : VWAP = ΣTradeValue / ΣVolume (₱/share)
    GSP_Mean     : mean daily Gross Sales Proceeds = TV_mu × P_avg
    mu_annual    : annualised log-return drift (full pre-CMEPA year)
    sig_annual   : annualised log-return volatility (full pre-CMEPA year)
    CAGR         : annualised price CAGR (full pre-CMEPA year)
    VOL_pre      : mean 5-day rolling volatility (Jul–Dec 2024 window)
    REV_BASELINE : Rev_B = GSP_Mean × STT_PRE × DAYS_6M (₱, 6-month basis)
    """
    _section_header(1, "ESTIMATED DISTRIBUTION PARAMETER CALCULATION")
    print("    Baseline window: Jul 2024 – Dec 2024 (seasonally aligned with "
          "post-CMEPA Jul–Dec 2025)")

    pre_full = pse_data['pre_cmepa'].copy()
    for col in ['Daily Returns', 'Closing Price', 'Trade Value', 'Volume']:
        pre_full[col] = pd.to_numeric(pre_full[col], errors='coerce')

    # 6-month aligned baseline
    pre = pre_full[(pre_full['Date'] >= '2024-07-01') &
                   (pre_full['Date'] <= '2024-12-31')].copy()
    if len(pre) == 0:
        raise ValueError("No data for Jul–Dec 2024. Check pre_cmepa sheet coverage.")

    vp = pre.dropna(subset=['Trade Value', 'Volume', 'Closing Price'])

    TV_mu    = float(vp['Volume'].mean())
    TV_sigma = float(vp['Volume'].std())
    P_avg    = (float(vp['Trade Value'].sum() / vp['Volume'].sum())
                if vp['Volume'].sum() > 0 else float(vp['Closing Price'].mean()))
    GSP_Mean = TV_mu * P_avg

    # Annualised log-return parameters (full pre-CMEPA year for stability)
    log_ret    = np.log(pre_full['Closing Price'] /
                        pre_full['Closing Price'].shift(1)).dropna()
    mu_annual  = float(log_ret.mean() * 252)
    sig_annual = float(log_ret.std()  * np.sqrt(252))

    # Annualised CAGR from daily price data
    price_start = float(pre_full['Closing Price'].iloc[0])
    price_end   = float(pre_full['Closing Price'].iloc[-1])
    n_yr        = len(pre_full) / 252
    CAGR        = (price_end / price_start) ** (1 / n_yr) - 1 if n_yr > 0 else 0.0

    VOL_pre      = float(pre['Volatility'].dropna().mean())
    REV_BASELINE = GSP_Mean * STT_PRE * DAYS_6M  # Rev_B on a 6-month basis

    # Full-period means for observed-change reporting (Stage 9)
    pre_tv_mean   = float(pse_data['pre_cmepa']['Trade Value'].mean())
    post_tv_mean  = float(pse_data['post_cmepa']['Trade Value'].mean())
    pre_vol_mean  = float(pse_data['pre_cmepa']['Volatility'].mean())
    post_vol_mean = float(pse_data['post_cmepa']['Volatility'].mean())

    print(f"\n    {'Parameter':<44} {'Value':>16}")
    print("    " + "─" * 62)
    print(f"    {'Mean Daily Share Volume (TV_µ)':<44} {TV_mu:>16,.2f} shares")
    print(f"    {'Std Dev Daily Share Volume (TV_σ)':<44} {TV_sigma:>16,.2f} shares")
    print(f"    {'VWAP (P_avg)':<44} ₱{P_avg:>14,.4f}")
    print(f"    {'Mean Daily GSP (TV_µ × P_avg)':<44} ₱{GSP_Mean:>14,.2f}")
    print(f"    {'5-Day Rolling Volatility Mean (VOL_pre)':<44} {VOL_pre:>16.6f}")
    print(f"    {'Annualised CAGR':<44} {CAGR*100:>15.4f}%")
    print(f"    {'6-Month Baseline Revenue (Rev_B)':<44} ₱{REV_BASELINE:>14,.0f}")
    print("    " + "─" * 62)
    print(f"    Rev_B = GSP_Mean × STT_PRE × DAYS_6M")
    print(f"          = ₱{GSP_Mean:,.2f} × {STT_PRE} × {DAYS_6M} = ₱{REV_BASELINE:,.0f}")

    return {
        'TV_mu': TV_mu,         'TV_sigma': TV_sigma,
        'P_avg': P_avg,         'GSP_Mean': GSP_Mean,
        'mu_annual': mu_annual, 'sig_annual': sig_annual,
        'CAGR': CAGR,           'VOL_pre': VOL_pre,
        'REV_BASELINE': REV_BASELINE,
        'pre_tv_mean':   pre_tv_mean,  'post_tv_mean':  post_tv_mean,
        'pre_vol_mean':  pre_vol_mean, 'post_vol_mean': post_vol_mean,
    }


# =============================================================================
# STAGE 2 — ELASTICITY PARAMETER ESTIMATION (TRAIN Law natural experiment)
# =============================================================================

def estimate_elasticity(pre_train: pd.DataFrame, post_train: pd.DataFrame) -> dict:
    """
    Estimates β_TV and β_VOL from the 2018 TRAIN Law period.

        β = (%ΔOutcome) / (%ΔSTT)
        %ΔSTT = TRAIN_STT_CHANGE = +20.0 %  (0.5 % → 0.6 %)

    β_TV is clipped at [−BETA_TV_CLIP, +BETA_TV_CLIP] to mitigate outlier influence.
    Both β parameters are sampled from Uniform(min, max) in the MCS.
    """
    _section_header(2, "ELASTICITY PARAMETER ESTIMATION (TRAIN Law, 2018)")
    print(f"    %ΔSTT (TRAIN Law) = +{TRAIN_STT_CHANGE*100:.0f}%  (0.5% → 0.6%)")
    print(f"    β_TV clipped at [−{BETA_TV_CLIP}, +{BETA_TV_CLIP}] to mitigate outlier influence")

    def _quarterly(df: pd.DataFrame) -> pd.DataFrame:
        return (df.copy()
                .assign(Quarter=lambda d: d['Date'].dt.to_period('Q'))
                .groupby('Quarter')
                .agg(volume_mean=('Trade Value', 'mean'),
                     volatility_sd=('Volatility', 'std'))
                .reset_index()
                .dropna())

    pre_q  = _quarterly(pre_train)
    post_q = _quarterly(post_train)
    n_q    = min(len(pre_q), len(post_q))

    beta_tv_raw, beta_vol_list = [], []
    for i in range(n_q):
        pct_tv = ((post_q.loc[i, 'volume_mean'] - pre_q.loc[i, 'volume_mean'])
                  / pre_q.loc[i, 'volume_mean'])
        beta_tv_raw.append(pct_tv / TRAIN_STT_CHANGE)

        pre_v  = pre_q.loc[i, 'volatility_sd']
        post_v = post_q.loc[i, 'volatility_sd']
        if pre_v and pre_v > 0:
            beta_vol_list.append((post_v - pre_v) / pre_v / TRAIN_STT_CHANGE)

    beta_tv_clipped = [np.clip(b, -BETA_TV_CLIP, BETA_TV_CLIP) for b in beta_tv_raw]

    results = {
        'BETA_TV_MIN':     float(min(beta_tv_clipped)),
        'BETA_TV_MAX':     float(max(beta_tv_clipped)),
        'BETA_TV_MEDIAN':  float(np.median(beta_tv_clipped)),
        'BETA_VOL_MIN':    float(min(beta_vol_list)),
        'BETA_VOL_MAX':    float(max(beta_vol_list)),
        'BETA_VOL_MEDIAN': float(np.median(beta_vol_list)),
    }

    print(f"\n    {'Parameter':<40} {'Min':>9}  {'Median':>9}  {'Max':>9}")
    print("    " + "─" * 70)
    print(f"    {'Volume Elasticity β_TV (clipped)':<40} "
          f"{results['BETA_TV_MIN']:>9.4f}  {results['BETA_TV_MEDIAN']:>9.4f}  "
          f"{results['BETA_TV_MAX']:>9.4f}")
    print(f"    {'Volatility Elasticity β_VOL':<40} "
          f"{results['BETA_VOL_MIN']:>9.4f}  {results['BETA_VOL_MEDIAN']:>9.4f}  "
          f"{results['BETA_VOL_MAX']:>9.4f}")
    print(f"    Quarterly pairs used: {n_q}")

    return results


# =============================================================================
# STAGE 3 — LAMBDA (λ) MEDIATING FACTOR DERIVATION [EDA Normal Sampling]
# =============================================================================

def calculate_lambda_ranges(pse_data: dict, med_data: dict) -> dict:
    """
    Derives EDA Normal distribution parameters (µ, σ) for each lambda component.
    All three lambdas are computed from empirical pre-CMEPA data only.

    λ₁  Participation Momentum Index (PMI):
            FLR_t / mean(FLR_pre)    — investor-type data

    λ₂  Intra-period Volatility Ratio (IVR):
            σ_t / mean(σ_pre)        — monthly return std devs from PSE data

    λ₃  Normalised Macro Pressure Index (NMPI):
            (GDP_t / GDP_pre_mean) × (π_pre_mean / π_t)
                                     — GDP growth and inflation data

    MCS sampling per iteration:
        λᵢ^(k) ~ Normal(µᵢ, σᵢ)   independently for i = 1, 2, 3
        λ^(k)  = λ₁^(k) × λ₂^(k) × λ₃^(k)
    """
    _section_header(3, "LAMBDA MEDIATING FACTOR DERIVATION [EDA Normal Sampling]")
    print("    All lambdas derived exclusively from pre-CMEPA data (Jul 2024 – Jun 2025).")

    # ── λ₁: Participation Momentum Index ─────────────────────────────────────
    # Normalise each monthly FLR observation by the pre-CMEPA period mean,
    # so the distribution of λ₁ = FLR_t / mean(FLR_pre) is centred near 1.
    flr     = med_data['flr'].copy()
    pre_flr = flr[flr['Period'] == 'Pre-CMEPA'].dropna(subset=['FLR']).copy()
    if len(pre_flr) == 0:
        raise ValueError("No Pre-CMEPA rows found in InvestorType_Data sheet.")
    flr_pre_mean     = float(pre_flr['FLR'].mean())
    lambda1_series   = pre_flr['FLR'] / flr_pre_mean
    lam1_mean        = float(lambda1_series.mean())   # ≈ 1.0 by construction
    lam1_std         = float(lambda1_series.std(ddof=1))
    print(f"\n    λ₁ (PMI): FLR_t / mean(FLR_pre={flr_pre_mean:.4f})")
    print(f"         µ = {lam1_mean:.4f}   σ = {lam1_std:.4f}  "
          f"(n = {len(pre_flr)} monthly observations)")

    # ── λ₂: Intra-period Volatility Ratio ────────────────────────────────────
    # Compute the monthly std dev of daily returns across the full pre-CMEPA year,
    # then normalise each month by the mean of those monthly std devs.
    pre_cm = pse_data['pre_cmepa'].copy()
    pre_cm['Daily Returns'] = pd.to_numeric(pre_cm['Daily Returns'], errors='coerce')
    pre_cm['Month']         = pre_cm['Date'].dt.to_period('M')
    pre_cm                  = pre_cm.dropna(subset=['Daily Returns'])
    sigma_monthly            = pre_cm.groupby('Month')['Daily Returns'].std()
    sigma_pre_mean           = float(sigma_monthly.mean())
    lambda2_series           = sigma_monthly / sigma_pre_mean
    lam2_mean                = float(lambda2_series.mean())   # ≈ 1.0 by construction
    lam2_std                 = float(lambda2_series.std(ddof=1))
    print(f"\n    λ₂ (IVR): σ_t / mean(σ_pre={sigma_pre_mean:.6f})")
    print(f"         µ = {lam2_mean:.4f}   σ = {lam2_std:.4f}  "
          f"(n = {len(lambda2_series)} monthly observations)")

    # ── λ₃: Normalised Macro Pressure Index ──────────────────────────────────
    # For each month in Jul 2024–Jun 2025, compute:
    #   λ₃_t = (GDP_t / GDP_pre_mean) × (π_pre_mean / π_t)
    # Matches the formula in Document 1 Stage 3.
    gdp = med_data['gdp'].copy()
    gdp['Year']       = pd.to_numeric(gdp['Year'],       errors='coerce')
    gdp['GDP_Growth'] = pd.to_numeric(gdp['GDP_Growth'], errors='coerce')
    gdp               = gdp.dropna(subset=['Year', 'GDP_Growth'])

    inf = med_data['inflation'].copy()
    inf['Year']          = pd.to_numeric(inf['Year'],          errors='coerce')
    inf['InflationRate'] = pd.to_numeric(inf['InflationRate'], errors='coerce')
    inf                  = inf.dropna(subset=['Year', 'InflationRate'])

    inf_pre = inf[inf['Classification'] == 'Pre-CMEPA'].reset_index(drop=True)
    if 'MonthNum' not in inf_pre.columns:
        _month_map       = {m: i + 1 for i, m in enumerate([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'])}
        inf_pre['MonthNum'] = inf_pre['Month'].map(_month_map)

    # Pre-CMEPA means — exclude projected Q3/Q4 2025 from GDP baseline
    gdp_pre      = gdp[~((gdp['Year'] == 2025) &
                         (gdp['Quarter'].isin(['Q3', 'Q4'])))].copy()
    GDP_PRE_MEAN = float(gdp_pre['GDP_Growth'].mean()) if len(gdp_pre) else float(gdp['GDP_Growth'].mean())
    PI_PRE_MEAN  = float(inf_pre['InflationRate'].mean()) if len(inf_pre) else float(inf['InflationRate'].mean())

    months_pre    = pd.date_range('2024-07-01', '2025-06-01', freq='MS')
    lambda3_vals  = []
    for m in months_pre:
        yr, mo  = m.year, m.month
        q_lbl   = f'Q{(mo - 1) // 3 + 1}'
        gdp_row = gdp[(gdp['Year'] == yr) & (gdp['Quarter'] == q_lbl)]['GDP_Growth']
        gdp_t   = float(gdp_row.iloc[0]) if len(gdp_row) else GDP_PRE_MEAN
        inf_row = inf_pre[(inf_pre['Year'] == yr) & (inf_pre['MonthNum'] == mo)]['InflationRate']
        pi_t    = float(inf_row.iloc[0]) if len(inf_row) else PI_PRE_MEAN
        pi_t    = max(pi_t, 1e-6)          # guard against division by zero
        lambda3_vals.append((gdp_t / GDP_PRE_MEAN) * (PI_PRE_MEAN / pi_t))

    lam3_arr  = np.array(lambda3_vals)
    lam3_mean = float(lam3_arr.mean())
    lam3_std  = float(lam3_arr.std(ddof=1))
    print(f"\n    λ₃ (NMPI): (GDP_t / GDP_pre_mean) × (π_pre_mean / π_t)")
    print(f"         GDP_pre_mean = {GDP_PRE_MEAN:.4f}   π_pre_mean = {PI_PRE_MEAN:.4f}")
    print(f"         µ = {lam3_mean:.4f}   σ = {lam3_std:.4f}  "
          f"(n = {len(lambda3_vals)} monthly observations)")

    lam_composite = lam1_mean * lam2_mean * lam3_mean
    print(f"\n    Composite λ_total (product of means): {lam_composite:.4f}")
    print(f"    Sampling in MCS: λᵢ^(k) ~ Normal(µᵢ, σᵢ); "
          f"λ^(k) = λ₁^(k) × λ₂^(k) × λ₃^(k)")

    return {
        'lam1_mean': lam1_mean, 'lam1_std': lam1_std,
        'lam2_mean': lam2_mean, 'lam2_std': lam2_std,
        'lam3_mean': lam3_mean, 'lam3_std': lam3_std,
    }


# =============================================================================
# STAGE 4 — TIME-HORIZON MONTE CARLO SIMULATION (FULL ITERATION STORAGE)
# =============================================================================

def _run_single_horizon(label: str, cfg: dict, eda_params: dict,
                        elasticity: dict, lambda_ranges: dict) -> dict:
    """
    Runs one time-horizon simulation with FULL ITERATION STORAGE.

    Revenue unit convention
    -----------------------
    flat_rev stores the PER-DAY single-day STT revenue (₱/day):
        rev_post_arr = gsp_post_arr × STT_POST      (₱ per day)

    6-month projected revenue is derived AFTER simulation:
        cumulative_rev = mean_rev_per_day × DAYS_6M  (₱ over 6 months)

    The Laffer test compares cumulative_rev (Rev_S) to REV_BASELINE (Rev_B),
    where Rev_B = GSP_Mean × STT_PRE × DAYS_6M — both on the same 6-month basis.

    Sensitivity analysis (Stage 6) correlates flat_beta_tv, flat_beta_vol,
    flat_lam, flat_tv_pre against flat_rev (daily revenue) — consistent because
    all are daily-level draws from the same simulation.
    """
    sims_per_day = cfg['sims_per_day']
    trading_days = cfg['trading_days']
    total_iters  = sims_per_day * trading_days

    TV_MEAN = eda_params['TV_mu']
    TV_SD   = eda_params['TV_sigma']
    P_AVG   = eda_params['P_avg']
    VOL_PRE = eda_params['VOL_pre']
    CAGR    = eda_params.get('CAGR', 0.0)
    REV_BASELINE = eda_params['REV_BASELINE']  # 6-month basis, pre-computed in Stage 1

    beta_tv_range  = (elasticity['BETA_TV_MIN'],  elasticity['BETA_TV_MAX'])
    beta_vol_range = (elasticity['BETA_VOL_MIN'], elasticity['BETA_VOL_MAX'])
    lam1_mean, lam1_std = lambda_ranges['lam1_mean'], lambda_ranges['lam1_std']
    lam2_mean, lam2_std = lambda_ranges['lam2_mean'], lambda_ranges['lam2_std']
    lam3_mean, lam3_std = lambda_ranges['lam3_mean'], lambda_ranges['lam3_std']

    print(f"\n    ── {label} Horizon ──")
    print(f"       Sims/day: {sims_per_day:,} | "
          f"Trading days: {trading_days:,} | "
          f"Total iterations: {total_iters:,}")
    print(f"       CAGR: {CAGR*100:.4f}% p.a.")

    # Allocate full storage arrays
    flat_tv_pct_all   = np.zeros(total_iters)
    flat_vol_pct_all  = np.zeros(total_iters)
    flat_rev_all      = np.zeros(total_iters)   # daily revenue (₱/day)
    flat_beta_tv_all  = np.zeros(total_iters)
    flat_beta_vol_all = np.zeros(total_iters)
    flat_lam_all      = np.zeros(total_iters)
    flat_tv_pre_all   = np.zeros(total_iters)

    idx = 0
    for d in range(trading_days):
        # Volume baseline grows with CAGR from the Jul–Dec 2024 anchor
        daily_growth = (1 + CAGR) ** (d / 252.0)

        tv_pre_arr   = np.random.normal(TV_MEAN * daily_growth, TV_SD, sims_per_day)
        tv_pre_arr   = np.maximum(tv_pre_arr, TV_MEAN * daily_growth * 0.05)

        beta_tv_arr  = np.random.uniform(*beta_tv_range,  sims_per_day)
        beta_vol_arr = np.random.uniform(*beta_vol_range, sims_per_day)

        lam1_arr = np.random.normal(lam1_mean, lam1_std, sims_per_day)
        lam2_arr = np.random.normal(lam2_mean, lam2_std, sims_per_day)
        lam3_arr = np.random.normal(lam3_mean, lam3_std, sims_per_day)
        lam_arr  = lam1_arr * lam2_arr * lam3_arr

        pct_tv_arr  = beta_tv_arr  * STT_CHANGE_PCT * lam_arr   # %ΔTV
        pct_vol_arr = beta_vol_arr * STT_CHANGE_PCT * lam_arr   # %ΔVOL

        tv_post_arr  = tv_pre_arr * (1 + pct_tv_arr)
        gsp_post_arr = tv_post_arr * P_AVG
        rev_post_arr = gsp_post_arr * STT_POST      # daily revenue (₱/day)

        end = idx + sims_per_day
        flat_tv_pct_all[idx:end]   = pct_tv_arr * 100
        flat_vol_pct_all[idx:end]  = pct_vol_arr * 100
        flat_rev_all[idx:end]      = rev_post_arr          # ₱/day
        flat_beta_tv_all[idx:end]  = beta_tv_arr
        flat_beta_vol_all[idx:end] = beta_vol_arr
        flat_lam_all[idx:end]      = lam_arr
        flat_tv_pre_all[idx:end]   = tv_pre_arr
        idx = end

    mean_tv_pct      = float(flat_tv_pct_all.mean())
    mean_vol_pct     = float(flat_vol_pct_all.mean())
    mean_rev_per_day = float(flat_rev_all.mean())
    # 6-month projected revenue — multiply mean daily by DAYS_6M
    cumulative_rev   = mean_rev_per_day * DAYS_6M

    # Laffer test on the 6-month basis (Rev_S vs Rev_B)
    laffer_ok = cumulative_rev > REV_BASELINE
    rev_chg   = (cumulative_rev - REV_BASELINE) / REV_BASELINE * 100

    print(f"\n       Mean % Δ Volume      : {mean_tv_pct:+.2f}%")
    print(f"       Mean % Δ Volatility  : {mean_vol_pct:+.2f}%")
    print(f"       Mean Daily Revenue   : ₱{mean_rev_per_day:,.2f}  (flat_rev unit: ₱/day)")
    print(f"       Rev_S (×{DAYS_6M} days)  : ₱{cumulative_rev:,.0f}")
    print(f"       Rev_B (baseline)     : ₱{REV_BASELINE:,.0f}")
    print(f"       Rev_S vs Rev_B       : {rev_chg:+.1f}%  → "
          f"{'FISCALLY ADEQUATE ✓' if laffer_ok else 'FISCALLY INADEQUATE ✗'}")

    return {
        'label':            label,
        'total_iters':      total_iters,
        'sims_per_day':     sims_per_day,
        'trading_days':     trading_days,
        'mean_tv_pct':      mean_tv_pct,
        'mean_vol_pct':     mean_vol_pct,
        'mean_rev_per_day': mean_rev_per_day,   # ₱/day
        'cumulative_rev':   cumulative_rev,      # ₱ over DAYS_6M (Rev_S)
        'REV_BASELINE':     REV_BASELINE,        # ₱ over DAYS_6M (Rev_B)
        'laffer_ok':        laffer_ok,
        'rev_chg_pct':      rev_chg,
        # Full daily-level arrays (unit: ₱/day for flat_rev; % for tv/vol)
        'flat_tv_pct':      flat_tv_pct_all,
        'flat_vol_pct':     flat_vol_pct_all,
        'flat_rev':         flat_rev_all,        # ₱/day — used in sensitivity
        'flat_beta_tv':     flat_beta_tv_all,
        'flat_beta_vol':    flat_beta_vol_all,
        'flat_lam':         flat_lam_all,
        'flat_tv_pre':      flat_tv_pre_all,
    }


def run_all_horizons(eda_params: dict, elasticity: dict, lambda_ranges: dict) -> dict:
    """Runs all three time-horizon simulations and prints consolidated results."""
    _section_header(4, "TIME-HORIZON MONTE CARLO SIMULATION (FULL STORAGE)")
    print(f"    Policy shock: STT {STT_PRE*100:.1f}% → {STT_POST*100:.1f}%"
          f"  ({STT_CHANGE_PCT*100:.2f}%)")
    print(f"    Model: %ΔY = β × (%Δt) × λ")

    horizon_results = {}
    for label, cfg in HORIZONS.items():
        horizon_results[label] = _run_single_horizon(
            label, cfg, eda_params, elasticity, lambda_ranges)

    # Consolidated table
    _subsection("Consolidated Results Across All Horizons")
    metrics_rows = []
    for metric_lbl, key, fmt in [
        ('Total Iterations',               'total_iters',      lambda v: f'{v:,}'),
        ('Mean % Δ Volume',                'mean_tv_pct',      lambda v: f'{v:+.2f}%'),
        ('Mean % Δ Volatility',            'mean_vol_pct',     lambda v: f'{v:+.2f}%'),
        ('Mean Daily Revenue (₱/day)',     'mean_rev_per_day', lambda v: f'₱{v:,.2f}'),
        ('6-Month Rev_S (mean×126d, ₱)',  'cumulative_rev',   lambda v: f'₱{v:,.0f}'),
        ('6-Month Rev_B (baseline, ₱)',   'REV_BASELINE',     lambda v: f'₱{v:,.0f}'),
        ('Rev_S vs Rev_B',                 'rev_chg_pct',      lambda v: f'{v:+.1f}%'),
        ('Laffer Verdict',                 'laffer_ok',        lambda v: 'ADEQUATE' if v else 'INADEQUATE'),
    ]:
        vals = [horizon_results[h][key] for h in HORIZONS]
        row  = [metric_lbl] + [fmt(v) for v in vals]
        metrics_rows.append(row)

    render_table(
        title="Table 4.  Consolidated Monte Carlo Results — All Time Horizons\n"
              "Model: %ΔY = β × (−83.33%) × λ  |  Full Iteration Storage",
        col_labels=['Metric', '1-Year\n(252,000)', '5-Year\n(630,000)', '10-Year\n(630,000)'],
        row_data=metrics_rows,
        fname='table4_horizon_results.png',
        col_widths=[0.40, 0.20, 0.20, 0.20],
        figsize=(14, 5),
        footnote=(
            f"Revenue unit: flat_rev stores daily single-day STT revenue (₱/day).  "
            f"Rev_S = mean_rev_per_day × {DAYS_6M} days (6-month projection).\n"
            f"Rev_B (baseline) = GSP_Mean × {STT_PRE} × {DAYS_6M} days — computed once in Stage 1 "
            f"and shared across all horizons.  Laffer test: Rev_S ≥ Rev_B."
        ),
        highlight_rows=[4, 5, 7],
    )

    return horizon_results


# =============================================================================
# STAGE 5 — EXPORT RESULTS TO EXCEL
# =============================================================================

def export_results_to_excel(horizon_results: dict, eda_params: dict,
                            elasticity: dict, lambda_ranges: dict) -> None:
    """
    Exports all simulation results to CMEPA_MCS_Results.xlsx.

    Sheets:
      Summary      — aggregate statistics per horizon (Rev_S, Rev_B, Laffer verdict)
      1Year        — all 252,000 iteration rows (daily revenue unit: ₱/day)
      5Year        — all 630,000 iteration rows
      10Year       — all 630,000 iteration rows
      Parameters   — all input parameters used in the simulation
    """
    _section_header(5, "EXPORTING RESULTS TO EXCEL")

    with pd.ExcelWriter('CMEPA_MCS_Results.xlsx', engine='openpyxl') as writer:

        # Sheet 1: Summary
        summary_rows = []
        for label in HORIZONS.keys():
            res = horizon_results[label]
            summary_rows.append({
                'Horizon':                   label,
                'Total Iterations':          res['total_iters'],
                'Sims Per Day':              res['sims_per_day'],
                'Trading Days':              res['trading_days'],
                'Mean % Δ Volume':           res['mean_tv_pct'],
                'Mean % Δ Volatility':       res['mean_vol_pct'],
                'Mean Daily Revenue (₱/day)':res['mean_rev_per_day'],
                'Rev_S — 6-Month (₱)':      res['cumulative_rev'],
                'Rev_B — 6-Month (₱)':      res['REV_BASELINE'],
                'Rev_S vs Rev_B (%)':        res['rev_chg_pct'],
                'Laffer Verdict':            'ADEQUATE' if res['laffer_ok'] else 'INADEQUATE',
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
        print("    [Summary sheet — aggregate stats per horizon]")

        # Sheets 2-4: Full iteration data
        for label in HORIZONS.keys():
            res = horizon_results[label]
            df_iters = pd.DataFrame({
                'Iteration':             np.arange(1, res['total_iters'] + 1),
                'TV_pct_Change_%':       res['flat_tv_pct'],
                'Vol_pct_Change_%':      res['flat_vol_pct'],
                # Column label clearly states ₱/day to match flat_rev unit
                'Daily_Revenue_PHP_day': res['flat_rev'],
                'Beta_TV':               res['flat_beta_tv'],
                'Beta_VOL':              res['flat_beta_vol'],
                'Lambda_Total':          res['flat_lam'],
                'TV_Pre_shares':         res['flat_tv_pre'],
            })
            sheet = label.replace('-', '').replace(' ', '')
            df_iters.to_excel(writer, sheet_name=sheet, index=False)
            print(f"    [{label:<8} sheet — {res['total_iters']:,} rows | "
                  f"Daily_Revenue_PHP_day unit: ₱/day]")

        # Sheet 5: Parameters
        pd.DataFrame({
            'Parameter': [
                'STT Pre-CMEPA (%)', 'STT Post-CMEPA (%)', 'STT Change (%)',
                'TRAIN STT Change (%)',
                'β_TV clip limit',
                'Days (6-month window)',
                'TV_µ (shares/day)', 'TV_σ (shares/day)', 'P_avg (₱/share)',
                'GSP_Mean (₱/day)', 'VOL_pre (5-day σ)', 'CAGR (%)',
                'REV_BASELINE / Rev_B (₱)',
                'β_TV_min', 'β_TV_median', 'β_TV_max',
                'β_VOL_min', 'β_VOL_median', 'β_VOL_max',
                'λ₁_mean', 'λ₁_std',
                'λ₂_mean', 'λ₂_std',
                'λ₃_mean', 'λ₃_std',
            ],
            'Value': [
                STT_PRE * 100, STT_POST * 100, STT_CHANGE_PCT * 100,
                TRAIN_STT_CHANGE * 100,
                BETA_TV_CLIP,
                DAYS_6M,
                eda_params['TV_mu'], eda_params['TV_sigma'], eda_params['P_avg'],
                eda_params['GSP_Mean'], eda_params['VOL_pre'], eda_params['CAGR'] * 100,
                eda_params['REV_BASELINE'],
                elasticity['BETA_TV_MIN'], elasticity['BETA_TV_MEDIAN'], elasticity['BETA_TV_MAX'],
                elasticity['BETA_VOL_MIN'], elasticity['BETA_VOL_MEDIAN'], elasticity['BETA_VOL_MAX'],
                lambda_ranges['lam1_mean'], lambda_ranges['lam1_std'],
                lambda_ranges['lam2_mean'], lambda_ranges['lam2_std'],
                lambda_ranges['lam3_mean'], lambda_ranges['lam3_std'],
            ],
            'Unit / Note': [
                '%', '%', '%  (fixed policy shock)',
                '%  (TRAIN Law STT increase)',
                'absolute β units',
                'trading days',
                'shares/day', 'shares/day', '₱/share',
                '₱/day', '5-day rolling σ', '% per annum',
                '₱ (GSP_Mean × STT_PRE × DAYS_6M)',
                'β units', 'β units', 'β units',
                'β units', 'β units', 'β units',
                'ratio', 'ratio',
                'ratio', 'ratio',
                'ratio', 'ratio',
            ],
        }).to_excel(writer, sheet_name='Parameters', index=False)
        print("    [Parameters sheet — all input constants and derived params]")

    print("    [Excel workbook saved → CMEPA_MCS_Results.xlsx]")


# =============================================================================
# STAGE 6 — SENSITIVITY ANALYSIS (Spearman's ρ + Tornado Chart)
# =============================================================================

def run_sensitivity_analysis(horizon_results: dict) -> dict:
    """
    Computes Spearman's Rank Correlation between stochastic inputs
    (β_TV, β_VOL, TV_pre, λ_total) and projected daily revenue (flat_rev, ₱/day)
    for each horizon.  All inputs and output are daily-level draws from the
    simulation — the unit of flat_rev (₱/day) is consistent with the inputs.

    Spearman's ρ interpretation:
      |ρ| ≥ 0.50  → Strong influence
      |ρ| 0.30–0.49 → Moderate influence
      |ρ| < 0.30  → Weak influence
    """
    _section_header(6, "SENSITIVITY ANALYSIS (Spearman's Rank Correlation)")

    all_sa = {}
    for label in HORIZONS.keys():
        res = horizon_results[label]

        inputs = {
            'Volume Elasticity (β_TV)':      res['flat_beta_tv'],
            'Volatility Elasticity (β_VOL)': res['flat_beta_vol'],
            'Baseline Trade Volume (TV_pre)': res['flat_tv_pre'],
            'Composite Lambda (λ_total)':    res['flat_lam'],
        }
        # Dependent variable: daily revenue (₱/day) — same level as all inputs
        output = res['flat_rev']

        rows = []
        for name, data in inputs.items():
            rho, pval = spearmanr(data, output)
            strength  = ('Strong'   if abs(rho) >= 0.50 else
                         'Moderate' if abs(rho) >= 0.30 else 'Weak')
            sig       = ('***' if pval < 0.001 else
                         '**'  if pval < 0.01  else
                         '*'   if pval < 0.05  else 'ns')
            rows.append({
                'Variable':  name,  'Rho':      rho,
                'AbsRho':    abs(rho), 'PValue': pval,
                'Sig':       sig,   'Strength': strength,
            })

        df_sa = (pd.DataFrame(rows)
                 .sort_values('AbsRho', ascending=False)
                 .reset_index(drop=True))
        all_sa[label] = df_sa

        print(f"\n    {label} (n = {res['total_iters']:,} iterations | "
              f"output: daily revenue ₱/day)")
        print(f"    {'Variable':<35} {'ρ':>9}  {'|ρ|':>6}  {'Sig':>5}  Influence")
        print("    " + "─" * 65)
        for _, row in df_sa.iterrows():
            print(f"    {row['Variable']:<35} {row['Rho']:>+9.4f}  "
                  f"{row['AbsRho']:>6.4f}  {row['Sig']:>5}  {row['Strength']}")

    # Tornado Chart — use 1-Year horizon for illustration
    _plot_tornado(all_sa, horizon_results)
    return all_sa


def _plot_tornado(all_sa: dict, horizon_results: dict) -> None:
    """Plots a three-panel tornado chart (one per horizon)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.patch.set_facecolor('white')
    fig.suptitle(
        "Figure — Tornado Chart: Spearman ρ — Sensitivity of Daily Revenue to Input Variables\n"
        "Longer bar = stronger monotonic association with projected daily STT revenue",
        fontsize=12, fontweight='bold', color=C_DARK)

    for ax, (label, df_sa) in zip(axes, all_sa.items()):
        ax.set_facecolor('white')
        bar_colors = [C_BLUE if r >= 0 else C_RED for r in df_sa['Rho']]
        bars = ax.barh(df_sa['Variable'], df_sa['Rho'],
                       color=bar_colors, edgecolor='#333333', linewidth=0.6,
                       height=0.48, alpha=0.85)
        ax.axvline(0, color=C_DARK, lw=1.0, ls='--', alpha=0.6)
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9, color=C_DARK)
        xlim = max(abs(df_sa['Rho'].min()), abs(df_sa['Rho'].max())) * 1.35
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Spearman ρ", fontsize=10)
        ax.set_title(f"{label} Horizon\n"
                     f"(n={horizon_results[label]['total_iters']:,})",
                     fontsize=10, fontweight='bold', color=C_DARK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('fig_tornado_sensitivity.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("\n    [Tornado chart saved → fig_tornado_sensitivity.png]")
    plt.show()


# =============================================================================
# STAGE 7 — MONTE CARLO DISTRIBUTION VISUALISATIONS
# =============================================================================

def plot_monte_carlo_distributions(horizon_results: dict) -> None:
    """
    Four-panel distribution chart using the 1-Year horizon.

    Panel D scatter subsamples 5,000 points for legibility
    (full 252,000-point scatter is unreadable at any alpha).
    """
    _section_header(7, "MONTE CARLO DISTRIBUTION VISUALISATIONS")

    res = horizon_results['1-Year']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f"Figure — Monte Carlo Output Distributions (1-Year Horizon, "
        f"n = {res['total_iters']:,} iterations)\n"
        f"Model: %ΔY = β × ({STT_CHANGE_PCT*100:.2f}%) × λ",
        fontsize=12, fontweight='bold', color=C_DARK)

    def _hist(ax, data, color, title, xlabel, fmt_str):
        mu  = data.mean()
        lo  = np.percentile(data, 2.5)
        hi  = np.percentile(data, 97.5)
        ax.set_facecolor('white')
        ax.hist(data, bins=60, color=color, alpha=0.55, edgecolor='none')
        ax.axvspan(lo, hi, alpha=0.10, color=color)
        ax.axvline(mu, color=C_DARK, lw=2.0, ls='-',
                   label=f'Mean: {fmt_str.format(mu)}')
        ax.axvline(lo, color=color, lw=1.4, ls='--',
                   label=f'95% CI: [{fmt_str.format(lo)}, {fmt_str.format(hi)}]')
        ax.axvline(hi, color=color, lw=1.4, ls='--')
        ax.text(0.97, 0.96,
                f"Mean: {fmt_str.format(mu)}\n"
                f"95% CI: [{fmt_str.format(lo)},\n"
                f"          {fmt_str.format(hi)}]\n"
                f"σ: {fmt_str.format(data.std())}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                          edgecolor='#444444', linewidth=0.7))
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', color=C_DARK)
        ax.legend(fontsize=8, frameon=True, edgecolor='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    _hist(axes[0, 0], res['flat_tv_pct'],
          C_BLUE, "Panel A — % Change in Trading Volume",
          "% Change in Trading Volume", "{:+.2f}%")

    _hist(axes[0, 1], res['flat_vol_pct'],
          C_RED,  "Panel B — % Change in Market Volatility",
          "% Change in 5-Day Volatility", "{:+.2f}%")

    _hist(axes[1, 0], res['flat_rev'],
          C_GREEN, "Panel C — Daily STT Revenue Distribution",
          "Daily Revenue (₱/day)", "₱{:,.0f}")

    # Panel D — Bivariate scatter (subsampled for legibility)
    ax_d = axes[1, 1]
    ax_d.set_facecolor('white')
    n_sub = 5_000
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(res['flat_tv_pct']), size=n_sub, replace=False)
    x_sub = res['flat_tv_pct'][idx]
    y_sub = res['flat_vol_pct'][idx]
    ax_d.scatter(x_sub, y_sub, alpha=0.25, s=8, color=C_BLUE)
    z = np.polyfit(x_sub, y_sub, 1)
    x_line = np.linspace(x_sub.min(), x_sub.max(), 200)
    ax_d.plot(x_line, np.poly1d(z)(x_line), color=C_RED, lw=2.0)
    corr = float(np.corrcoef(x_sub, y_sub)[0, 1])
    ax_d.text(0.05, 0.95,
              f"Pearson r = {corr:.4f}\n(subsample n = {n_sub:,})",
              transform=ax_d.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=C_LIGHT))
    ax_d.set_xlabel("% Change in Trading Volume", fontsize=9)
    ax_d.set_ylabel("% Change in Volatility", fontsize=9)
    ax_d.set_title("Panel D — Bivariate Relationship\n%ΔTV vs. %ΔVOL (5,000-point subsample)",
                   fontsize=10, fontweight='bold', color=C_DARK)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('fig_monte_carlo_distributions.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("    [Figure saved → fig_monte_carlo_distributions.png]")
    plt.show()


# =============================================================================
# STAGE 8 — POLICY SHOCK TRANSMISSION FLOWCHART
# =============================================================================

def plot_policy_shock_flowchart() -> None:
    """
    Policy transmission flowchart showing how the STT shock propagates
    through lambda mediators and elasticities to trading volume, volatility,
    and revenue.

    Each box uses a white fill with a coloured border so text remains
    readable and borders are visible.
    """
    _section_header(8, "POLICY SHOCK TRANSMISSION FLOWCHART")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    fig.suptitle(
        "Figure — Policy Shock Transmission: STT 0.6% → 0.1% (−83.33%)\n"
        "Pathway: Policy Shock → Mediating Factors → Elasticity → Market Outcomes",
        fontsize=11, fontweight='bold', color=C_DARK)

    def _box(ax, x, y, w, h, text, border_color, fill_color='white',
             text_color=None, fontsize=9):
        """Draws a box with a visible coloured border and readable interior text."""
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle='round,pad=0.12',
            facecolor=fill_color,
            edgecolor=border_color,
            linewidth=2.2,
            zorder=3)
        ax.add_patch(rect)
        tc = text_color if text_color else border_color
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=tc, fontfamily='serif',
                multialignment='center', zorder=4)

    def _arrow(ax, x1, y1, x2, y2, color=C_DARK):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=color,
                            lw=2.0, mutation_scale=18), zorder=5)

    # Row 1 — Policy shock
    _box(ax, 5, 9.3, 3.2, 0.75,
         'POLICY SHOCK\nCMEPA: STT 0.6% → 0.1%  (−83.33%)',
         border_color=C_RED, fill_color='#fdf0f0', text_color=C_RED)

    # Arrows down to lambda row
    for xdest in [1.5, 5.0, 8.5]:
        _arrow(ax, 5, 8.92, xdest, 8.45, color=C_DARK)

    # Row 2 — Lambda mediators
    for x, lbl, desc in [
        (1.5, 'λ₁  Investor Type',  'Participation\nMomentum Index\nFLR_t / mean(FLR)'),
        (5.0, 'λ₂  Market Conditions','Intra-period\nVolatility Ratio\nσ_t / mean(σ)'),
        (8.5, 'λ₃  Macro Factors',  'Normalised Macro\nPressure Index\n(GDP/GDP₀)×(π₀/π)'),
    ]:
        _box(ax, x, 7.75, 2.4, 1.20, f'{lbl}\n{desc}',
             border_color=C_BLUE, fill_color='#eef2f8', text_color=C_BLUE, fontsize=8.2)

    # Arrows converging to composite lambda
    for xsrc in [1.5, 5.0, 8.5]:
        _arrow(ax, xsrc, 7.15, 5.0, 6.65, color=C_BLUE)

    # Row 3 — Composite lambda
    _box(ax, 5, 6.25, 3.8, 0.70,
         'Composite  λ = λ₁ × λ₂ × λ₃\n'
         'Each λᵢ ~ Normal(µᵢ, σᵢ)  [EDA Sampling]',
         border_color=C_BLUE, fill_color='#dde4f0', text_color=C_NAVY, fontsize=8.5)

    # Arrow down to model equation
    _arrow(ax, 5, 5.90, 5, 5.40, color=C_DARK)

    # Row 4 — Elasticity application
    for x, lbl in [(2.5, 'β_TV ~ Uniform(β_min, β_max)\nVolume Elasticity'),
                   (7.5, 'β_VOL ~ Uniform(β_min, β_max)\nVolatility Elasticity')]:
        _box(ax, x, 5.05, 3.0, 0.65, lbl,
             border_color=C_AMBER, fill_color='#fdf6e3', text_color=C_AMBER, fontsize=8.2)
    _arrow(ax, 2.5, 4.72, 3.5, 4.35, color=C_AMBER)
    _arrow(ax, 7.5, 4.72, 6.5, 4.35, color=C_AMBER)

    # Row 5 — Core model equations
    _box(ax, 5, 4.05, 6.0, 0.75,
         '%ΔTV  = β_TV  × (−83.33%) × λ\n'
         '%ΔVOL = β_VOL × (−83.33%) × λ',
         border_color=C_DARK, fill_color='#f0f0f0', text_color=C_DARK, fontsize=9)

    # Arrows to outcomes
    _arrow(ax, 3.0, 3.67, 2.0, 3.15, color=C_GREEN)
    _arrow(ax, 7.0, 3.67, 8.0, 3.15, color=C_RED)

    # Row 6 — Market outcomes
    _box(ax, 2.0, 2.75, 2.6, 0.70,
         'Trading Volume\nTV_post = TV_pre × (1 + %ΔTV)',
         border_color=C_GREEN, fill_color='#eaf4ec', text_color=C_GREEN, fontsize=8.2)
    _box(ax, 8.0, 2.75, 2.6, 0.70,
         'Market Volatility\nVOL_post = VOL_pre × (1 + %ΔVOL)',
         border_color=C_RED, fill_color='#fdf0f0', text_color=C_RED, fontsize=8.2)

    # Arrows to revenue
    _arrow(ax, 2.0, 2.40, 4.0, 1.75, color=C_GREEN)
    _arrow(ax, 8.0, 2.40, 6.0, 1.75, color=C_GREEN)

    # Row 7 — Revenue
    _box(ax, 5, 1.40, 4.5, 0.65,
         'Rev_S = GSP_post × 0.1% × DAYS_6M\n'
         'Laffer Test: Rev_S ≥ Rev_B?',
         border_color=C_GREEN, fill_color='#eaf4ec', text_color=C_GREEN, fontsize=9)

    _arrow(ax, 5, 1.07, 5, 0.60, color=C_DARK)

    # Row 8 — Verdict
    _box(ax, 5, 0.30, 4.5, 0.55,
         'Policy Verdict: Fiscally Adequate / Inadequate',
         border_color=C_NAVY, fill_color='#e8ecf4', text_color=C_NAVY, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('fig_policy_shock_flowchart.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("    [Figure saved → fig_policy_shock_flowchart.png]")
    plt.show()


# =============================================================================
# STAGE 9 — FINAL RESEARCH SUMMARY
# =============================================================================

def print_final_summary(horizon_results: dict, eda_params: dict,
                        elasticity: dict, lambda_ranges: dict) -> None:
    """Prints a structured consolidated research summary."""
    _section_header(9, "FINAL RESEARCH SUMMARY")

    obs_tv_chg  = ((eda_params['post_tv_mean']  - eda_params['pre_tv_mean'])
                   / eda_params['pre_tv_mean']  * 100)
    obs_vol_chg = ((eda_params['post_vol_mean'] - eda_params['pre_vol_mean'])
                   / eda_params['pre_vol_mean'] * 100)
    lam_prod    = (lambda_ranges['lam1_mean']
                   * lambda_ranges['lam2_mean']
                   * lambda_ranges['lam3_mean'])

    print(f"\n    {'Finding':<56} {'Value':>14}")
    print("    " + "─" * 72)
    print(f"    {'Observed Volume Change (actual PSE data)':<56} {obs_tv_chg:>+13.1f}%")
    print(f"    {'Observed Volatility Change (actual PSE data)':<56} {obs_vol_chg:>+13.1f}%")
    print(f"    {'β_TV median (volume elasticity)':<56} {elasticity['BETA_TV_MEDIAN']:>+13.4f}")
    print(f"    {'β_VOL median (volatility elasticity)':<56} {elasticity['BETA_VOL_MEDIAN']:>+13.4f}")
    print(f"    {'λ_total composite (λ₁×λ₂×λ₃ at means)':<56} {lam_prod:>+13.4f}")

    for label, res in horizon_results.items():
        print(f"\n    ── {label} Horizon ({res['total_iters']:,} iterations) ──")
        print(f"    {'Mean % Δ Volume':<56} {res['mean_tv_pct']:>+13.2f}%")
        print(f"    {'Mean % Δ Volatility':<56} {res['mean_vol_pct']:>+13.2f}%")
        print(f"    {'Mean Daily Revenue (₱/day)':<56} ₱{res['mean_rev_per_day']:>12,.2f}")
        print(f"    {'6-Month Rev_S (mean × 126 days)':<56} ₱{res['cumulative_rev']:>12,.0f}")
        print(f"    {'6-Month Rev_B (baseline)':<56} ₱{res['REV_BASELINE']:>12,.0f}")
        print(f"    {'Rev_S vs Rev_B':<56} {res['rev_chg_pct']:>+13.1f}%")
        verdict = 'ADEQUATE ✓' if res['laffer_ok'] else 'INADEQUATE ✗'
        print(f"    {'Laffer Fiscal Adequacy Verdict':<56} {verdict:>14}")

    print("\n    " + "─" * 72)
    print("\n    Output files generated:")
    print("      table4_horizon_results.png")
    print("      CMEPA_MCS_Results.xlsx  (Summary | 1Year | 5Year | 10Year | Parameters)")
    print("      fig_tornado_sensitivity.png")
    print("      fig_monte_carlo_distributions.png")
    print("      fig_policy_shock_flowchart.png")
    print("\n" + "="*78)
    print("    SIMULATION COMPLETE")
    print("="*78)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':

    print("="*78)
    print("  CMEPA MONTE CARLO SIMULATION — PRODUCTION VERSION")
    print("  Full Iteration Storage | All Horizons | Excel Export")
    print("  Bajo, Cacho, Rizon, Villamor, Ylaya | USJ-R | BSA 2025–2026")
    print("="*78)
    print(f"\n  Horizon Structure:")
    for lbl, cfg in HORIZONS.items():
        total = cfg['sims_per_day'] * cfg['trading_days']
        print(f"    {lbl:<10}: {cfg['sims_per_day']:>5,} sims/day × "
              f"{cfg['trading_days']:>5,} trading days = {total:>9,} iterations")

    # Stage 0 — Load data
    pse_data = load_pse_data(PSE_FILE)
    med_data = load_mediating_data(MED_FILE)

    # Stage 1 — EDA parameters (incl. REV_BASELINE)
    eda_params = calculate_eda_parameters(pse_data)

    # Stage 2 — Elasticity
    elasticity = estimate_elasticity(pse_data['pre_train'], pse_data['post_train'])

    # Stage 3 — Lambda (all three derived from data, none hardcoded)
    lambda_ranges = calculate_lambda_ranges(pse_data, med_data)

    # Stage 4 — Monte Carlo simulation (all horizons, full storage)
    horizon_results = run_all_horizons(eda_params, elasticity, lambda_ranges)

    # Stage 5 — Export to Excel
    export_results_to_excel(horizon_results, eda_params, elasticity, lambda_ranges)

    # Stage 6 — Sensitivity analysis
    sensitivity = run_sensitivity_analysis(horizon_results)

    # Stage 7 — Distribution visualisations
    plot_monte_carlo_distributions(horizon_results)

    # Stage 8 — Policy shock flowchart
    plot_policy_shock_flowchart()

    # Stage 9 — Final summary
    print_final_summary(horizon_results, eda_params, elasticity, lambda_ranges)