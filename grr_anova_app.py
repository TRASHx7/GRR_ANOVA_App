"""
GRR ANOVA Viewer
================
A Tkinter desktop application for performing Gauge Repeatability & Reproducibility (GRR)
analysis using two-way ANOVA.

Workflow:
  1. Load a CSV with columns: operator, slot, part, tp1 [, tp2, ...]
  2. Select a measurement column (tp column) from the dropdown.
  3. Click "Run ANOVA" to fit two-way ANOVA models for all three factor pairs:
       operator x part, operator x slot, slot x part
  4. View interaction/main-effects plots and the ANOVA table / variance component breakdown.

CSV format expected:
  - operator : categorical identifier for the person who took the measurement
  - slot     : categorical identifier for the measurement slot/position
  - part     : categorical identifier for the part being measured
  - tp1, tp2, ... : one or more numeric measurement (test-point) columns
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")  # Use the Tkinter-compatible Matplotlib backend
import matplotlib.pyplot  # used by on_export_table to render DataFrames as images
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import f as f_dist  # used to recompute p-values for random-effects F-tests


# Columns that must be present in every CSV loaded into the app.
# All other columns are treated as measurement (tp) columns.
REQUIRED_ID_COLS = ["operator", "slot", "part"]


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names.

    Prevents hard-to-debug mismatches caused by invisible spaces in CSV headers.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _validate_csv_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the DataFrame is missing required columns or has too few columns.

    Ensures the CSV has at least operator, slot, part, and one measurement column
    before any analysis is attempted.
    """
    missing = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected at least {REQUIRED_ID_COLS}.")
    if len(df.columns) < 4:
        raise ValueError("CSV must have at least 4 columns: operator, slot, part, and at least one tp column.")


def measurement_columns(df: pd.DataFrame) -> list[str]:
    """Return all column names that are not part of the required ID columns.

    These are the measurement (tp) columns the user can select for analysis.
    """
    return [c for c in df.columns if c not in REQUIRED_ID_COLS]


def coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all required ID columns to string type.

    Ensures operator, slot, and part are treated as categorical factors rather
    than numbers, which is important for ANOVA formula construction.
    """
    out = df.copy()
    for c in REQUIRED_ID_COLS:
        out[c] = out[c].astype(str)
    return out


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric, turning unparseable values into NaN.

    Non-numeric entries (e.g. text in a measurement column) are silently dropped
    downstream via dropna().
    """
    return pd.to_numeric(s, errors="coerce")


def fit_two_way_anova(df: pd.DataFrame, ycol: str, a: str, b: str, interaction: bool):
    """Fit a two-way OLS ANOVA model and return the ANOVA table.

    Parameters
    ----------
    df          : Full dataset (must contain columns a, b, and ycol).
    ycol        : Name of the numeric measurement column (the response variable).
    a, b        : Names of the two categorical factor columns (e.g. "operator", "part").
    interaction : If True, include the A*B interaction term in the model.

    Returns
    -------
    model     : Fitted OLS model object from statsmodels.
    anova_tbl : Type-II ANOVA table as a DataFrame (columns: sum_sq, df, F, PR(>F)).
    d         : Cleaned subset of df used for fitting (rows with NaN removed).

    Notes
    -----
    Q('name') quoting in the formula allows column names containing spaces or
    special characters to be used safely with statsmodels patsy formulas.
    C() wraps each factor to ensure it is treated as categorical.
    """
    d = df[[a, b, ycol]].copy()
    d[a] = d[a].astype(str)
    d[b] = d[b].astype(str)
    d[ycol] = coerce_numeric_series(d[ycol])
    d = d.dropna()

    if d.empty:
        raise ValueError(f"No valid numeric data for {ycol} after cleaning.")

    # Build patsy formula string: with or without the A:B interaction term
    if interaction:
        formula = f"Q('{ycol}') ~ C(Q('{a}')) + C(Q('{b}')) + C(Q('{a}')):C(Q('{b}'))"
    else:
        formula = f"Q('{ycol}') ~ C(Q('{a}')) + C(Q('{b}'))"

    model = smf.ols(formula=formula, data=d).fit()
    anova_tbl = anova_lm(model, typ=2)  # Type II sums of squares
    return model, anova_tbl, d


def _fix_random_effects_f(anova_tbl: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Recompute F-statistics and p-values for the two main effects using random-effects denominators.

    statsmodels OLS divides every term by MS_E (fixed-effects assumption), which
    inflates F and produces over-optimistic p-values for the main effects in a
    random-effects GRR study. The correct denominators are:

      F(A)   = MS_A  / MS_AB  (df1 = df_A,  df2 = df_AB)
      F(B)   = MS_B  / MS_AB  (df1 = df_B,  df2 = df_AB)
      F(A*B) = MS_AB / MS_E   (unchanged — already correct)

    The function operates on the raw anova_tbl (patsy index strings) returned by
    anova_lm before any renaming. Returns a corrected copy; the original is unchanged.
    If MS_AB == 0 the corrected F and p-values are set to NaN.
    """
    tbl = anova_tbl.copy()

    row_a  = [idx for idx in tbl.index if str(idx).startswith(f"C(Q('{a}'))") and ":" not in str(idx)]
    row_b  = [idx for idx in tbl.index if str(idx).startswith(f"C(Q('{b}'))") and ":" not in str(idx)]
    row_ab = [idx for idx in tbl.index if ":" in str(idx)]

    if not row_a or not row_b or not row_ab:
        return tbl  # interaction term missing (additive model); nothing to fix

    ms_a  = float(tbl.loc[row_a[0],  "sum_sq"] / tbl.loc[row_a[0],  "df"])
    ms_b  = float(tbl.loc[row_b[0],  "sum_sq"] / tbl.loc[row_b[0],  "df"])
    ms_ab = float(tbl.loc[row_ab[0], "sum_sq"] / tbl.loc[row_ab[0], "df"])

    df_a  = float(tbl.loc[row_a[0],  "df"])
    df_b  = float(tbl.loc[row_b[0],  "df"])
    df_ab = float(tbl.loc[row_ab[0], "df"])

    if ms_ab > 0:
        f_a = ms_a / ms_ab
        f_b = ms_b / ms_ab
        # Survival function: P(F > observed) = the p-value
        p_a = float(f_dist.sf(f_a, df_a, df_ab))
        p_b = float(f_dist.sf(f_b, df_b, df_ab))
    else:
        f_a = f_b = p_a = p_b = float("nan")

    tbl.loc[row_a[0], "F"]      = f_a
    tbl.loc[row_b[0], "F"]      = f_b
    tbl.loc[row_a[0], "PR(>F)"] = p_a
    tbl.loc[row_b[0], "PR(>F)"] = p_b

    return tbl


def _prettify_anova_index(anova_tbl: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Return a copy of anova_tbl with human-readable row labels.

    Replaces patsy term strings (e.g. "C(Q('operator'))") with plain names.
    Only used for display — the raw table is still passed to variance_components_balanced,
    which relies on the original patsy strings to locate rows.

    Mapping applied:
      C(Q('{a}'))               -> a          (e.g. "operator")
      C(Q('{b}'))               -> b          (e.g. "part")
      anything containing ":"   -> "{a} x {b}"  (the interaction term)
      Residual                  -> Residual   (unchanged)
    """
    rename = {}
    for idx in anova_tbl.index:
        s = str(idx)
        if ":" in s:
            rename[idx] = f"{a} x {b}"
        elif s.startswith(f"C(Q('{a}'))"):
            rename[idx] = a
        elif s.startswith(f"C(Q('{b}'))"):
            rename[idx] = b
    return anova_tbl.rename(index=rename)


def is_balanced_two_way(df: pd.DataFrame, a: str, b: str) -> tuple[bool, str]:
    """Check whether the two-way design is balanced (equal replicates in every cell).

    Variance component estimation using the Mean Squares method (see
    variance_components_balanced) is only valid for balanced designs. An unbalanced
    design requires more complex REML/restricted-ML estimation not implemented here.

    Returns (True, "") if balanced, or (False, reason_string) if not.
    """
    counts = df.groupby([a, b]).size()
    if counts.empty:
        return False, "No cells available."
    if (counts == counts.iloc[0]).all():
        return True, ""
    return False, "Unbalanced cells. Replicates per A x B cell are not constant."


def variance_components_balanced(anova_tbl: pd.DataFrame, df: pd.DataFrame, a: str, b: str) -> dict:
    """Estimate variance components from a balanced two-way random-effects ANOVA.

    Uses the classical Mean Squares (MS) method, which is only valid when every
    A x B cell has the same number of replicates (n).

    Formulas (random-effects model):
      Var(A)        = (MS_A  - MS_AB) / (b_levels * n)
      Var(B)        = (MS_B  - MS_AB) / (a_levels * n)
      Var(A*B)      = (MS_AB - MS_E)  / n
      Var(E)        = MS_E              (pure repeatability / within-cell error)

    Negative variance estimates (which can occur when MS differences are negative
    due to sampling variability) are clipped to 0, following standard GRR practice.

    Returns a dict with keys: a, b, "{a}*{b}", "Repeatability", "Total".
    """
    cell_counts = df.groupby([a, b]).size()
    n = int(cell_counts.iloc[0])           # replicates per cell (balanced, so all equal)
    a_levels = df[a].nunique()
    b_levels = df[b].nunique()

    # Locate the correct rows in the ANOVA table by matching the patsy term strings.
    # The interaction row contains ":" (e.g. "C(Q('operator')):C(Q('part'))").
    row_a  = [idx for idx in anova_tbl.index if str(idx).startswith(f"C(Q('{a}'))") and ":" not in str(idx)]
    row_b  = [idx for idx in anova_tbl.index if str(idx).startswith(f"C(Q('{b}'))") and ":" not in str(idx)]
    row_ab = [idx for idx in anova_tbl.index if ":" in str(idx)]
    if not row_a or not row_b or not row_ab or "Residual" not in anova_tbl.index:
        raise ValueError("ANOVA table does not contain expected terms for variance component computation.")

    # Compute Mean Squares: MS = SS / df  (statsmodels stores SS, not MS directly)
    ms_a  = float(anova_tbl.loc[row_a[0],  "sum_sq"] / anova_tbl.loc[row_a[0],  "df"])
    ms_b  = float(anova_tbl.loc[row_b[0],  "sum_sq"] / anova_tbl.loc[row_b[0],  "df"])
    ms_ab = float(anova_tbl.loc[row_ab[0], "sum_sq"] / anova_tbl.loc[row_ab[0], "df"])
    ms_e  = float(anova_tbl.loc["Residual","sum_sq"] / anova_tbl.loc["Residual","df"])

    # Apply MS method formulas
    var_a  = (ms_a  - ms_ab) / (b_levels * n)
    var_b  = (ms_b  - ms_ab) / (a_levels * n)
    var_ab = (ms_ab - ms_e)  / n
    var_e  = ms_e

    comps = {
        a:               max(0.0, var_a),
        b:               max(0.0, var_b),
        f"{a}*{b}":      max(0.0, var_ab),
        "Repeatability": max(0.0, var_e),   # within-cell measurement noise
    }
    comps["Total"] = sum(comps.values())
    return comps


def contribution_percentages(components: dict) -> dict:
    """Convert variance components to percentage contributions relative to the total.

    Each value in the returned dict is (component / total) * 100.
    Returns zeros for all components if the total is zero or negative (degenerate case).
    The "Total" key from the input dict is excluded from the output.
    """
    total = components.get("Total", 0.0)
    if total <= 0:
        return {k: 0.0 for k in components if k != "Total"}
    return {k: 100.0 * v / total for k, v in components.items() if k != "Total"}


def interaction_means_plot(fig: Figure, df: pd.DataFrame, ycol: str, a: str, b: str, title: str) -> None:
    """Draw an interaction plot: cell means of ycol with A levels on the x-axis, B levels as lines.

    Non-parallel lines indicate a potential A*B interaction effect — i.e. the effect
    of factor A on the measurement depends on the level of factor B.

    Parameters
    ----------
    fig   : Matplotlib Figure to draw into.
    df    : Cleaned dataset (already validated and numeric).
    ycol  : Measurement column name (y-axis).
    a, b  : Factor column names; A goes on the x-axis, B becomes separate lines.
    title : Plot title string.
    """
    ax = fig.add_subplot(111)
    ax.clear()

    means = df.groupby([a, b])[ycol].mean().reset_index()
    a_levels = sorted(df[a].unique().tolist())
    b_levels = sorted(df[b].unique().tolist())

    x = np.arange(len(a_levels))
    for b_lvl in b_levels:
        # Extract cell means for this level of B, aligned to the full list of A levels
        sub = means[means[b] == b_lvl].set_index(a).reindex(a_levels)
        y = sub[ycol].to_numpy()
        ax.plot(x, y, marker="o", label=str(b_lvl))

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in a_levels], rotation=30, ha="right")
    ax.set_xlabel(a)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(title=b, fontsize=8, title_fontsize=9)
    fig.tight_layout()


def main_effects_plot(fig: Figure, df: pd.DataFrame, ycol: str, a: str, b: str, title: str) -> None:
    """Draw a main effects plot showing marginal means for each factor.

    Both factors are overlaid on the same axes using an integer level index on the
    x-axis (since A and B may have different numbers of levels). A legend text box
    at the bottom-left maps each index back to the original level label.

    This plot corresponds to the no-interaction (additive) model and shows the
    individual effect of each factor on the measurement.

    Parameters
    ----------
    fig   : Matplotlib Figure to draw into.
    df    : Cleaned dataset.
    ycol  : Measurement column name (y-axis).
    a, b  : Factor column names.
    title : Plot title string.
    """
    ax = fig.add_subplot(111)
    ax.clear()

    # Compute marginal means: average ycol over all levels of the other factor
    a_means = df.groupby(a)[ycol].mean().sort_index()
    b_means = df.groupby(b)[ycol].mean().sort_index()

    ax.plot(np.arange(len(a_means)), a_means.values, marker="o", label=f"Means by {a}")
    ax.plot(np.arange(len(b_means)), b_means.values, marker="o", label=f"Means by {b}")

    ax.set_title(title)
    ax.set_xlabel("Level index (see legend)")
    ax.set_ylabel(ycol)
    ax.legend(fontsize=8)

    # Build a compact index-to-label mapping shown as an annotation box
    a_map = ", ".join([f"{i}:{lvl}" for i, lvl in enumerate(a_means.index.tolist())])
    b_map = ", ".join([f"{i}:{lvl}" for i, lvl in enumerate(b_means.index.tolist())])
    ax.text(
        0.01, 0.01,
        f"{a} levels: {a_map}\n{b} levels: {b_map}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()


def contribution_bar_plot(fig: Figure, pct: dict, title: str) -> None:
    """Draw a bar chart of variance component percentage contributions.

    Each bar represents one variance source (operator, part, interaction, repeatability).
    A percentage label is printed above each bar. The y-axis is fixed at 0–100%.

    Parameters
    ----------
    fig   : Matplotlib Figure to draw into.
    pct   : Dict mapping component name -> % contribution (from contribution_percentages).
    title : Plot title string.
    """
    ax = fig.add_subplot(111)
    ax.clear()

    labels = list(pct.keys())
    values = [pct[k] for k in labels]
    x = np.arange(len(labels))

    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("% contribution")
    ax.set_ylim(0, 100)
    ax.set_title(title)

    # Annotate the exact percentage above each bar
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()


class GRRAnovaApp(tk.Tk):
    """Main application window for the GRR ANOVA Viewer.

    Layout (top to bottom):
      - Toolbar: Load CSV button, measurement column selector, Run ANOVA button, status bar.
      - Vertical paned window split between:
          Top pane  — Tabbed plot notebook (interaction plots, main effects, contribution bars).
          Bottom pane — Table selector combobox + tabbed table notebook (ANOVA table, variance components).

    State:
      self.df              : The currently loaded DataFrame.
      self.anova_tables    : Dict of ANOVA result DataFrames, keyed by display label.
      self.component_tables: Dict of variance component DataFrames, keyed by the same labels.
    """

    def __init__(self):
        super().__init__()
        self.title("ANOVA GRR Viewer")
        self.geometry("1200x820")

        self.df: pd.DataFrame | None = None
        self.file_path: str | None = None

        # Results caches populated by on_run(); keyed by the same label strings
        # that populate the "Bottom table view" combobox.
        self.anova_tables: dict[str, pd.DataFrame] = {}
        self.component_tables: dict[str, pd.DataFrame] = {}

        # Matplotlib Figure objects keyed by plot title (for export).
        self.plot_figures: dict[str, Figure] = {}
        # FigureCanvasTkAgg widgets keyed by plot title (for show/hide swapping).
        self.plot_canvases: dict[str, FigureCanvasTkAgg] = {}

        self._build_ui()

    def _build_ui(self):
        """Construct and lay out all widgets.

        Called once from __init__. Widgets are stored as instance attributes so
        that event handlers can read/modify them later.
        """
        # --- Toolbar row ---
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Button(top, text="Load CSV", command=self.on_load_csv).pack(side="left")

        ttk.Label(top, text="Measurement (tp column):").pack(side="left", padx=(15, 5))
        self.tp_var = tk.StringVar(value="")
        # Combobox is disabled until a CSV is loaded; populated with tp column names on load
        self.tp_combo = ttk.Combobox(top, textvariable=self.tp_var, state="disabled", width=30)
        self.tp_combo.pack(side="left")

        # Run button is disabled until a CSV is successfully loaded
        self.btn_run = ttk.Button(top, text="Run ANOVA", command=self.on_run, state="disabled")
        self.btn_run.pack(side="left", padx=(15, 0))

        # Status bar (single line below the toolbar)
        self.status_var = tk.StringVar(value="Load a CSV to begin.")
        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=10)

        # --- Vertical paned split: plots (top, weight=3) / tables (bottom, weight=2) ---
        self.paned = ttk.Panedwindow(self, orient="vertical")
        self.paned.pack(fill="both", expand=True, padx=10, pady=10)

        plot_frame = ttk.Frame(self.paned)
        self.paned.add(plot_frame, weight=3)

        table_frame = ttk.Frame(self.paned)
        self.paned.add(table_frame, weight=2)

        # --- Plot area: selector row + canvas display frame ---
        plot_top = ttk.Frame(plot_frame)
        plot_top.pack(fill="x", pady=(0, 4))

        ttk.Label(plot_top, text="Plot:").pack(side="left")
        self.plot_select_var = tk.StringVar(value="")
        # Dropdown to switch between available plots; populated after each ANOVA run
        self.plot_select = ttk.Combobox(plot_top, textvariable=self.plot_select_var,
                                        state="disabled", width=55)
        self.plot_select.pack(side="left", padx=(8, 8))
        self.plot_select.bind("<<ComboboxSelected>>", self.on_plot_selected)

        # Exports whichever plot is currently selected in the dropdown
        self.btn_export_plot = ttk.Button(plot_top, text="Export Plot PNG",
                                          command=self.on_export_plot, state="disabled")
        self.btn_export_plot.pack(side="right", padx=(0, 4))

        # Canvas area: all plot canvases are created here; only one is visible at a time
        self.plot_canvas_frame = ttk.Frame(plot_frame)
        self.plot_canvas_frame.pack(fill="both", expand=True)

        # --- Bottom section: selector combobox + ANOVA/components tabs ---
        table_top = ttk.Frame(table_frame)
        table_top.pack(fill="x", pady=(0, 6))

        ttk.Label(table_top, text="Bottom table view:").pack(side="left")
        self.table_select_var = tk.StringVar(value="")
        # Lists every factor-pair combination run; switching updates both table tabs
        self.table_select = ttk.Combobox(table_top, textvariable=self.table_select_var, state="disabled", width=55)
        self.table_select.pack(side="left", padx=(8, 8))
        self.table_select.bind("<<ComboboxSelected>>", self.on_table_selected)

        # Exports whichever table tab (ANOVA or Components) is currently visible
        self.btn_export_table = ttk.Button(table_top, text="Export Table PNG",
                                           command=self.on_export_table, state="disabled")
        self.btn_export_table.pack(side="right", padx=(0, 4))

        # Optional note label (e.g. balance warnings)
        self.table_note_var = tk.StringVar(value="")
        ttk.Label(table_top, textvariable=self.table_note_var).pack(side="left", padx=(8, 0))

        self.bottom_nb = ttk.Notebook(table_frame)
        self.bottom_nb.pack(fill="both", expand=True)

        # Tab 1: ANOVA Table (sum_sq, df, F, p-value)
        self.anova_tab = ttk.Frame(self.bottom_nb)
        self.bottom_nb.add(self.anova_tab, text="ANOVA Table")

        self.anova_tree = ttk.Treeview(self.anova_tab, show="headings")
        self.anova_tree.pack(side="left", fill="both", expand=True)

        anova_vsb = ttk.Scrollbar(self.anova_tab, orient="vertical", command=self.anova_tree.yview)
        anova_vsb.pack(side="right", fill="y")
        self.anova_tree.configure(yscrollcommand=anova_vsb.set)

        # Tab 2: Variance Components (variance estimate + % contribution per source)
        self.comp_tab = ttk.Frame(self.bottom_nb)
        self.bottom_nb.add(self.comp_tab, text="Variance Components")

        self.comp_tree = ttk.Treeview(self.comp_tab, show="headings")
        self.comp_tree.pack(side="left", fill="both", expand=True)

        comp_vsb = ttk.Scrollbar(self.comp_tab, orient="vertical", command=self.comp_tree.yview)
        comp_vsb.pack(side="right", fill="y")
        self.comp_tree.configure(yscrollcommand=comp_vsb.set)

        # Single shared horizontal scrollbar — forwards scroll to whichever tab is active
        self.hsb = ttk.Scrollbar(table_frame, orient="horizontal")
        self.hsb.pack(fill="x")
        self.hsb.configure(command=self._xview_active_tree)

        # Both trees report their x-position to the shared scrollbar
        self.anova_tree.configure(xscrollcommand=self.hsb.set)
        self.comp_tree.configure(xscrollcommand=self.hsb.set)

        # Reset scroll position when switching between the two table tabs
        self.bottom_nb.bind("<<NotebookTabChanged>>", lambda _e: self._sync_hscroll())

    def _active_tree(self) -> ttk.Treeview:
        """Return the Treeview widget in the currently visible bottom tab."""
        tab_text = self.bottom_nb.tab(self.bottom_nb.select(), "text")
        return self.comp_tree if tab_text == "Variance Components" else self.anova_tree

    def _xview_active_tree(self, *args):
        """Forward horizontal scroll commands from the shared scrollbar to the active tree."""
        self._active_tree().xview(*args)

    def _sync_hscroll(self):
        """Reset horizontal scroll to position 0 when the user switches table tabs.

        Without this the scrollbar thumb position from the previous tab would persist,
        which looks confusing if the two tables have different widths.
        """
        self._active_tree().xview_moveto(0)

    def set_status(self, msg: str):
        """Update the status bar text."""
        self.status_var.set(msg)

    def on_load_csv(self):
        """Open a file dialog, load the selected CSV, and populate the tp column selector.

        Validates the schema before accepting the file. On success, enables the
        measurement combobox and Run button. On failure, shows an error dialog.
        """
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return  # User cancelled the dialog
        try:
            df = pd.read_csv(path)
            df = _normalize_cols(df)        # strip whitespace from headers
            _validate_csv_schema(df)        # ensure required columns are present
            df = coerce_categoricals(df)    # cast operator/slot/part to str

            tps = measurement_columns(df)
            if not tps:
                raise ValueError("No tp columns found. Expected tp1, tp2, ..., tpn after operator/slot/part.")

            self.df = df
            self.file_path = path

            # Populate and enable the measurement selector
            self.tp_combo["values"] = tps
            self.tp_combo["state"] = "readonly"
            self.tp_combo.current(0)
            self.btn_run["state"] = "normal"

            self.set_status(f"Loaded CSV. Rows: {len(df)}. Pick a tp column and run ANOVA.")
            self.table_note_var.set("")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.set_status("Failed to load CSV.")

    def _clear_plot_tabs(self):
        """Destroy all plot canvases, reset the dropdown, and disable the export button."""
        for canvas in self.plot_canvases.values():
            canvas.get_tk_widget().destroy()
        self.plot_canvases.clear()
        self.plot_figures.clear()
        self.plot_select["values"] = []
        self.plot_select_var.set("")
        self.plot_select["state"] = "disabled"
        self.btn_export_plot["state"] = "disabled"

    def _add_plot_tab(self, title: str, fig: Figure):
        """Create a canvas for the figure, hide it, and add its title to the dropdown.

        The canvas is packed into plot_canvas_frame but immediately hidden with
        pack_forget. on_plot_selected shows the chosen canvas and hides all others.
        """
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.get_tk_widget().pack_forget()  # hidden until selected

        self.plot_figures[title] = fig
        self.plot_canvases[title] = canvas

        # Append to dropdown values
        current = list(self.plot_select["values"]) if self.plot_select["values"] else []
        current.append(title)
        self.plot_select["values"] = current
        self.plot_select["state"] = "readonly"
        self.btn_export_plot["state"] = "normal"

        # Auto-select the first plot added
        if len(current) == 1:
            self.plot_select_var.set(title)
            canvas.get_tk_widget().pack(fill="both", expand=True)

    def on_plot_selected(self, *_):
        """Show the canvas for the selected plot and hide all others."""
        selected = self.plot_select_var.get()
        for title, canvas in self.plot_canvases.items():
            widget = canvas.get_tk_widget()
            if title == selected:
                widget.pack(fill="both", expand=True)
            else:
                widget.pack_forget()

    def _tree_set_df(self, tree: ttk.Treeview, df: pd.DataFrame, float_fmt: str = "{:.6g}"):
        """Populate a Treeview with the contents of a DataFrame.

        The DataFrame's index is inserted as the first "Source" column so that
        row identifiers (ANOVA term names, component names) are always visible.
        Numeric columns are formatted with float_fmt for compact display.

        Parameters
        ----------
        tree      : The Treeview widget to populate.
        df        : DataFrame whose rows become Treeview rows.
        float_fmt : Python format string applied to every float/int cell.
        """
        # Remove all existing rows and column definitions
        tree.delete(*tree.get_children())
        tree["columns"] = ()

        d = df.copy()
        d.insert(0, "Source", d.index.astype(str))  # make index a visible column

        # Format numeric values for readability
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]) or pd.api.types.is_integer_dtype(d[c]):
                d[c] = d[c].apply(lambda x: float_fmt.format(x) if pd.notna(x) else "")

        cols = list(d.columns)
        tree["columns"] = cols

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="center", stretch=True)

        for _, row in d.iterrows():
            tree.insert("", "end", values=list(row.values))

    def _set_current_bottom_view(self, key: str):
        """Load the ANOVA table and variance components tables for the given key.

        key is a label like "operator x part (with interaction)". If either table
        is missing for the key, the corresponding Treeview is cleared.
        """
        # Update ANOVA table tab
        if key in self.anova_tables:
            self._tree_set_df(self.anova_tree, self.anova_tables[key])
        else:
            self.anova_tree.delete(*self.anova_tree.get_children())
            self.anova_tree["columns"] = ()

        # Update Variance Components tab
        if key in self.component_tables:
            self._tree_set_df(self.comp_tree, self.component_tables[key], float_fmt="{:.6g}")
        else:
            self.comp_tree.delete(*self.comp_tree.get_children())
            self.comp_tree["columns"] = ()

    def on_table_selected(self, _event=None):
        """Called when the user changes the selection in the bottom-table combobox."""
        key = self.table_select_var.get()
        if not key:
            return
        self._set_current_bottom_view(key)
        self.table_note_var.set("")

    def on_export_plot(self):
        """Save the currently selected plot as a PNG file.

        Opens a Save-As dialog pre-filled with the plot's title as the filename.
        Uses the stored Matplotlib Figure to write the PNG (dpi=150 for a crisp result).
        """
        title = self.plot_select_var.get()
        if not title:
            messagebox.showwarning("Export", "No plot is currently displayed.")
            return
        fig = self.plot_figures.get(title)
        if fig is None:
            messagebox.showwarning("Export", "Figure not found for the current tab.")
            return

        # Suggest a filename derived from the tab title (replace special chars with _)
        safe_name = title.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        path = filedialog.asksaveasfilename(
            title="Save plot as PNG",
            initialfile=f"{safe_name}.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return  # User cancelled
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.set_status(f"Plot saved: {path}")

    def on_export_table(self):
        """Save the currently visible table tab (ANOVA or Variance Components) as a PNG.

        Renders the DataFrame using Matplotlib's table renderer so the output is a
        self-contained image file rather than a CSV. Opens a Save-As dialog.
        """
        key = self.table_select_var.get()
        if not key:
            messagebox.showwarning("Export", "No results are loaded. Run ANOVA first.")
            return

        # Decide which DataFrame to export based on the active bottom tab
        tab_text = self.bottom_nb.tab(self.bottom_nb.select(), "text")
        if tab_text == "Variance Components":
            df_raw = self.component_tables.get(key)
            label = "components"
        else:
            df_raw = self.anova_tables.get(key)
            label = "anova"

        if df_raw is None or df_raw.empty:
            messagebox.showwarning("Export", "The current table has no data to export.")
            return

        # Build a display copy with the index as the first "Source" column
        d = df_raw.copy()
        d.insert(0, "Source", d.index.astype(str))
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]) or pd.api.types.is_integer_dtype(d[c]):
                d[c] = d[c].apply(lambda x: f"{x:.6g}" if pd.notna(x) else "")

        path = filedialog.asksaveasfilename(
            title="Save table as PNG",
            initialfile=f"{label}_{key.replace(' ', '_')}.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return  # User cancelled

        # Render the DataFrame as a Matplotlib table and save
        n_rows, n_cols = d.shape
        fig_h = max(1.5, 0.4 * (n_rows + 1))  # scale height with number of rows
        fig_w = max(4.0, 1.6 * n_cols)
        export_fig, ax = matplotlib.pyplot.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText=d.values.tolist(),
            colLabels=list(d.columns),
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(list(range(n_cols)))
        ax.set_title(f"{tab_text} — {key}", fontsize=10, pad=10)
        export_fig.tight_layout()
        export_fig.savefig(path, dpi=150, bbox_inches="tight")
        matplotlib.pyplot.close(export_fig)  # free memory; don't display in the GUI
        self.set_status(f"Table saved: {path}")

    def on_run(self):
        """Entry point for running the ANOVA analysis on the selected measurement column.

        Clears previous results, then runs _run_pair for each of the three factor
        combinations (operator x part, operator x slot, slot x part). Results are
        stored in self.anova_tables and self.component_tables and the first result
        is shown automatically.
        """
        if self.df is None:
            return

        ycol = self.tp_var.get()
        if not ycol:
            messagebox.showwarning("Select tp", "Pick a measurement column first.")
            return

        self._clear_plot_tabs()

        # Reset result caches from any previous run
        self.anova_tables.clear()
        self.component_tables.clear()

        # The three factor pairs covering all combinations of operator, slot, and part
        pairs = [("operator", "part"), ("operator", "slot"), ("slot", "part")]

        self.set_status(f"Running ANOVA for {ycol}...")
        self.table_note_var.set("")

        for a, b in pairs:
            self._run_pair(ycol, a, b)

        # Populate the table selector combobox with results
        keys = list(self.anova_tables.keys())
        if keys:
            self.table_select["values"] = keys
            self.table_select["state"] = "readonly"
            self.table_select.current(0)
            self.table_select_var.set(keys[0])
            self._set_current_bottom_view(keys[0])
            self.btn_export_table["state"] = "normal"
        else:
            self.table_select["values"] = []
            self.table_select_var.set("")
            self.table_select["state"] = "disabled"
            self.btn_export_table["state"] = "disabled"

        self.set_status(f"Done. Showing results for {ycol}.")

    def _run_pair(self, ycol: str, a: str, b: str):
        """Run both ANOVA models (with and without interaction) for one factor pair.

        Generates up to four plot tabs per pair:
          1. Interaction means plot  (with-interaction model)
          2. % Contribution bar chart (with-interaction model, balanced data only)
          3. Main effects plot       (no-interaction / additive model)

        Results are stored in self.anova_tables and self.component_tables under
        two keys: "{a} x {b} (with interaction)" and "{a} x {b} (no interaction)".

        Errors for either model are caught individually so that a failure in one
        model does not prevent the other from running.
        """
        # --- Model 1: with A*B interaction term ---
        try:
            _model_i, anova_i, df_clean = fit_two_way_anova(self.df, ycol, a, b, interaction=True)

            key_i = f"{a} x {b} (with interaction)"
            # Correct F/p for main effects before display (raw table still used for variance components)
            anova_i_fixed = _fix_random_effects_f(anova_i, a, b)
            self.anova_tables[key_i] = _prettify_anova_index(anova_i_fixed, a, b)

            fig1 = Figure(figsize=(7, 5), dpi=100)
            interaction_means_plot(fig1, df_clean, ycol, a, b, title=f"{ycol}: Interaction plot ({a} x {b})")
            self._add_plot_tab(key_i, fig1)

            # Variance components can only be estimated for balanced designs
            balanced, reason = is_balanced_two_way(df_clean, a, b)
            if balanced:
                comps = variance_components_balanced(anova_i, df_clean, a, b)
                pct = contribution_percentages(comps)

                # Build a combined DataFrame with variance and % contribution columns
                comp_df = pd.DataFrame(
                    {
                        "Variance": {k: v for k, v in comps.items() if k != "Total"},
                        "% Contribution": pct,
                    }
                )
                self.component_tables[key_i] = comp_df

                figc = Figure(figsize=(7, 5), dpi=100)
                contribution_bar_plot(figc, pct, title=f"{ycol}: % contribution ({a} x {b})")
                self._add_plot_tab(f"{a} x {b} (% contrib)", figc)
            else:
                # Design is unbalanced; store a note instead of variance components
                comp_df = pd.DataFrame({"Note": [reason]}, index=["Components not computed"])
                self.component_tables[key_i] = comp_df

        except Exception as e:
            # Store the error message as a single-row DataFrame so the table view
            # still shows something meaningful instead of an empty widget
            key_i = f"{a} x {b} (with interaction)"
            err_df = pd.DataFrame({"Error": [str(e)]}, index=["Failed"])
            self.anova_tables[key_i] = err_df
            self.component_tables[key_i] = pd.DataFrame({"Error": [str(e)]}, index=["Failed"])

        # --- Model 2: additive model (no interaction term) ---
        try:
            _model_n, anova_n, df_clean2 = fit_two_way_anova(self.df, ycol, a, b, interaction=False)

            key_n = f"{a} x {b} (no interaction)"
            # No interaction row exists in the additive model, so _fix_random_effects_f
            # returns the table unchanged — called here for consistency and future-proofing.
            anova_n_fixed = _fix_random_effects_f(anova_n, a, b)
            self.anova_tables[key_n] = _prettify_anova_index(anova_n_fixed, a, b)
            # Variance components are only computed for the interaction model;
            # show an informational note here instead
            self.component_tables[key_n] = pd.DataFrame(
                {"Note": ["Variance components are tied to the interaction model view."]},
                index=["Info"]
            )

            fig2 = Figure(figsize=(7, 5), dpi=100)
            main_effects_plot(fig2, df_clean2, ycol, a, b, title=f"{ycol}: Main effects ({a}, {b})")
            self._add_plot_tab(key_n, fig2)

        except Exception as e:
            key_n = f"{a} x {b} (no interaction)"
            err_df = pd.DataFrame({"Error": [str(e)]}, index=["Failed"])
            self.anova_tables[key_n] = err_df
            self.component_tables[key_n] = pd.DataFrame({"Error": [str(e)]}, index=["Failed"])


if __name__ == "__main__":
    app = GRRAnovaApp()
    app.mainloop()
