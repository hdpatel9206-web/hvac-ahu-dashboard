"""
patch_app.py
============
Patches app_updated.py in place, after taking a timestamped backup.

PERFORMANCE
  1  SHAP_SAMPLE_CAP 300 -> 120
  2  adds cached_shap_values() so the Top-N slider stops recomputing SHAP
  3  page 3 uses the cache, with check_additivity=False

DATA CORRECTIONS  (values that currently contradict the thesis)
  4  learning curve: fabricated monotonic array -> real CSV values
  5  adaptation curve: rejected-variant figures -> finetune_fixed_results.json
  6  cites run_20260409_022716_ASHRAE_LBNL instead of run_20260415_142209
  7  binary cross-building F1 0.628 -> 0.415
  8  per-class valve/damper F1 0.98 -> 1.0000
  9  top SHAP feature SA_Temp_std_30min 0.0378 -> Return Air Damper 0.0932
 10  adversarial trials 1,000 -> 600 ; verification runtime 3.02s -> 1.37s
 11  Days 24-26 split by day (Day 24 four-class, Day 25 binary peak)
 12  labels ZSSC figures as surrogate-classifier output
 13  finetune_results.json -> finetune_fixed_results.json

Run:  python patch_app.py
"""
import ast
import os
import shutil
import sys
import time

TARGET = "app_updated.py"

P = []          # (label, find, replace)


def add(label, find, replace):
    P.append((label, find, replace))


# ── 1. sample cap ────────────────────────────────────────────────────────
add("SHAP_SAMPLE_CAP 300 -> 120",
    "SHAP_SAMPLE_CAP = 300",
    "SHAP_SAMPLE_CAP = 120   # TreeSHAP cost is linear in sample count")

# ── 2. cached SHAP wrapper ───────────────────────────────────────────────
add("add cached_shap_values()",
    'def load_explainer(_model):\n    import shap\n    return shap.TreeExplainer(_model)',
    'def load_explainer(_model):\n'
    '    import shap\n'
    '    return shap.TreeExplainer(_model)\n'
    '\n'
    '\n'
    '@st.cache_data(show_spinner="Computing SHAP values (first run only)...")\n'
    'def cached_shap_values(_explainer, X):\n'
    '    """Cached so the Top-N slider does not trigger recomputation.\n'
    '\n'
    '    The leading underscore tells Streamlit not to hash the explainer.\n'
    '    check_additivity=False skips a second full verification pass.\n'
    '    """\n'
    '    return _explainer.shap_values(X, check_additivity=False)')

# ── 3. use the cache on page 3 ───────────────────────────────────────────
add("page 3 uses cached SHAP",
    '    with st.spinner("Computing SHAP values\u2026"):\n'
    '        sv = explainer.shap_values(X_sample)',
    '    sv = cached_shap_values(explainer, X_sample)')

# ── 4. learning curve ────────────────────────────────────────────────────
add("learning curve -> real CSV values",
    '    f1_lc  = [0.498, 0.511, 0.524, 0.531, 0.538, 0.543, 0.547, 0.551, 0.555, 0.557, 0.559]',
    '    # verified from results/learning_curve_results.csv, f1_cb column (macro-F1)\n'
    '    f1_lc  = [0.5463, 0.5473, 0.5155, 0.5348, 0.5472,\n'
    '              0.5317, 0.5309, 0.5306, 0.4983, 0.5224, 0.5592]')

add("learning curve caption",
    'st.caption("F1 stays flat 0.498\u20130.559 across all 11 training proportions. "',
    'st.caption("Macro-F1 stays flat 0.4983\u20130.5592 across all 11 proportions "\n'
    '               "(mean 0.5331, SD 0.0171, fitted slope \u22120.0100 per unit proportion). "')

add("learning curve source file",
    '               "Source: learning_curve_results.json.")',
    '               "Source: results/learning_curve_results.csv.")')

# ── 5. adaptation curve ──────────────────────────────────────────────────
add("adaptation curve data",
    '    pcts   = [0, 1, 5, 10, 15, 20, 30, 50]\n'
    '    f1_4cl = [0.331, 0.916, 0.977, 0.992, 0.994, 0.997, 0.998, 0.999]',
    '    # verified from results/finetune_fixed_results.json (f1_4 and f1_2)\n'
    '    pcts   = [0, 5, 10, 15, 20, 30, 50]\n'
    '    f1_4cl = [0.3116, 0.7395, 0.7440, 0.7468, 0.7474, 0.7485, 0.7498]\n'
    '    f1_bin = [0.4154, 0.9860, 0.9920, 0.9958, 0.9965, 0.9980, 0.9997]')

add("adaptation curve plot",
    '    ax.axhline(0.331, color="#e74c3c", ls="--", lw=1.2, label="Baseline (0.331)")\n'
    '    ax.axvline(10,    color="#f39c12", ls="--", lw=1.2, label="10% threshold")',
    '    ax.plot(pcts, f1_bin, "s-", color="#27ae60", lw=2, ms=6, label="Binary F1")\n'
    '    ax.axhline(0.75, color="#7f8c8d", ls=":", lw=1.4,\n'
    '               label="Theoretical max 0.75 (Class 1 absent)")\n'
    '    ax.axvline(10,   color="#f39c12", ls="--", lw=1.2, label="10% recommended")')

add("adaptation curve annotation",
    '    ax.annotate("0.992 at 10%", xy=(10, 0.992), xytext=(18, 0.88),',
    '    ax.annotate("0.7440 = 99.2% of ceiling", xy=(10, 0.7440), xytext=(20, 0.58),')

add("adaptation caption",
    '        "Verified from finetune_results.json."',
    '        "4-class macro-F1 rises 0.3116 \u2192 0.7440, which is 99.2% of the 0.75 "\n'
    '        "ceiling imposed by the absence of Class 1. "\n'
    '        "Verified from results/finetune_fixed_results.json."')

# ── 6. correct the cited run ─────────────────────────────────────────────
add("PSI source note -> primary run",
    '        "Source: Original model PSI values from run_20260415_142209/psi_results.json. "',
    '        "Source: results/run_20260409_022716_ASHRAE_LBNL/results_summary.json "\n'
    '        "(psi_top_features field). "')

add("fault-analysis fallback -> primary run",
    'run_20260415_142209):**',
    'run_20260409_022716_ASHRAE_LBNL):**')

# ── 7 & 8. per-class and binary figures ──────────────────────────────────
add("per-class F1 0.98 -> 1.0000",
    '            "- All four fault classes: Healthy F1=1.00, Sensor Bias F1=1.00, "\n'
    '            "Valve Fault F1=0.98, Damper Fault F1=0.98"',
    '            "- Per-class in-sample F1: Healthy 0.9828, Sensor Bias 0.9973, "\n'
    '            "Valve Fault 1.0000, Damper Fault 1.0000"')

add("binary cross-building 0.628 -> 0.415",
    '        "Binary F1": [0.999, 0.628, 0.992, "N/A"],',
    '        "Binary F1": [0.999, 0.415, 0.9920, "N/A"],')

add("cross-building summary caption",
    '    st.caption("Source: Self-created from verified JSON experiment files.")',
    '    st.caption(\n'
    '        "Source: results_summary.json and finetune_fixed_results.json. "\n'
    '        "Binary cross-building F1 = 0.415 for the primary model; the 0.628 figure "\n'
    '        "belongs to the optimised variant rejected in Section 4.14."\n'
    '    )')

# ── 9. top feature ───────────────────────────────────────────────────────
add("page 3 header caption",
    '        "Top feature: SA_Temp_std_30min = 0.0378"',
    '        "Top feature: AHU Return Air Damper Control Signal"')

add("page 3 info callout",
    '        "**Top SHAP feature:** Supply Air Temperature 30-minute rolling standard deviation "\n'
    '        "(SA_Temp_rs30) \u2014 mean |SHAP| = **0.0378**. "\n'
    '        "This reflects the physical fault signature: valve and coil faults manifest as "\n'
    '        "increased supply air temperature variability over time, not instantaneous spikes."',
    '        "**Top-ranked feature:** AHU Return Air Damper Control Signal, impurity "\n'
    '        "importance **0.0932**, and first by SHAP attribution on the same model artefact. "\n'
    '        "It is a control command rather than a measurement: it encodes how this building "\n'
    '        "was commissioned to manage its outdoor-air fraction. Ranks 2 to 5 are all outdoor "\n'
    '        "air temperature features. Every one of the top five sits on a channel the PSI gate "\n'
    '        "classifies RED, which is precisely the mechanism of cross-building failure."')

add("sensor-bias recommendation",
    '        "SHAP top driver: SA_Temp_std_30min = 0.0378."',
    '        "Top attribution drivers: outdoor air temperature features (PSI 2.45, RED gate)."')

add("valve-fault cost interpretation",
    '            "SHAP top feature SA_Temp_std_30min (0.0378) reflects thermal instability "\n'
    '            "consistent with valve fault mechanism."',
    '            "Supply air temperature variability and the coil temperature difference "\n'
    '            "(Temp_supply_return_diff) are among the strongest attribution features, "\n'
    '            "consistent with the valve fault mechanism."')

# ── 10. verification counts ──────────────────────────────────────────────
add("SLSQP trials 1,000 -> 600",
    '                "100% (1,000/1,000 trials)",',
    '                "100% (600/600 trials)",')

add("verification runtime 3.02 -> 1.37",
    '                "3.02 seconds",',
    '                "1.37 seconds",')

add("CBF code comment trials",
    '# RESULT: 0 violations across 1,000 adversarial trials',
    '# RESULT: 0 violations across 600 adversarial trials (3 states x 200)')

# ── 11. Days 24-26 by day ────────────────────────────────────────────────
add("Days 24-26 metrics",
    '    gc1.metric("Gate State",      "GREEN \u2705")\n'
    '    gc2.metric("Binary F1",       f"{m[\'green_binary\']:.3f}", "+27.9pp vs baseline")',
    '    gc1.metric("Gate State", "GREEN \u2705", "Days 24\u201326")\n'
    '    gc2.metric("Binary F1 peak", "0.6098", "Day 25")')

# ── 12. surrogate disclosure ─────────────────────────────────────────────
add("safety result scope",
    '        "**This is a mathematical guarantee, not a probabilistic result.**"',
    '        "**The barrier function is a mathematical constraint, not a probabilistic "\n'
    '        "result** \u2014 and it was additionally verified on real BMS data: 0 violations "\n'
    '        "across 203,040 timesteps from five datasets in South Korea and Ireland."')

add("ZSSC surrogate warning",
    '    st.info(\n'
    '        "**Why F1 = 0.336 is NOT a failure:** "',
    '    st.warning(\n'
    '        "**These classifier figures come from a calibrated surrogate, not the trained "\n'
    '        "model.** A closed-loop experiment requires the classifier to respond to setpoint "\n'
    '        "changes the controller itself generates, which stored data cannot provide. The "\n'
    '        "surrogate is parameterised to reproduce the measured anchors of this study "\n'
    '        "(about 0.99 under zero shift, 0.415 binary F1 at full shift). The CBF safety "\n'
    '        "result is unaffected, because the barrier function constrains actuation rather "\n'
    '        "than classification. See Section 5.5."\n'
    '    )\n'
    '    st.info(\n'
    '        "**Why F1 = 0.336 is NOT a failure:** "')

# ── 13. PSI threshold caption ────────────────────────────────────────────
add("PSI gate caption",
    '        "AMBER = PSI 0.10\u20130.50 (caution) \u00b7 "',
    '        "AMBER = 0.10\u20130.50 (collect adaptation data) \u00b7 "')


def main():
    if not os.path.exists(TARGET):
        print("!! " + TARGET + " not found. Run this from the thesis folder.")
        sys.exit(1)

    src = open(TARGET, encoding="utf-8").read()
    original = src

    backup = TARGET + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(TARGET, backup)
    print("Backup: " + backup + "\n")

    applied, failed = 0, []
    for label, find, repl in P:
        c = src.count(find)
        if c == 1:
            src = src.replace(find, repl)
            applied += 1
            print("  ok    " + label)
        elif c == 0:
            failed.append((label, "not found"))
            print("  MISS  " + label)
        else:
            failed.append((label, str(c) + " matches, skipped"))
            print("  SKIP  " + label + "  (" + str(c) + " matches)")

    if src == original:
        print("\nNothing matched. File untouched.")
        os.remove(backup)
        return

    try:
        ast.parse(src)
    except SyntaxError as e:
        print("\n!! Patched result does not parse: line %s: %s" % (e.lineno, e.msg))
        print("!! Nothing written. Original intact, backup at " + backup)
        sys.exit(1)

    open(TARGET, "w", encoding="utf-8").write(src)
    print("\n%d of %d patches applied. Syntax check passed." % (applied, len(P)))
    print("Written: " + TARGET)

    if failed:
        print("\nNot applied \u2014 patch these by hand:")
        for label, why in failed:
            print("  - %s  (%s)" % (label, why))

    print("\nNext:  python -m streamlit run app_updated.py")
    print("First SHAP load still takes a few minutes; afterwards it is cached.")


if __name__ == "__main__":
    main()
