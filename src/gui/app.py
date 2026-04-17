import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.simulation.traffic_model import build_single_intersection, build_multi_intersection
from src.optimization.pso import PSO, PSOConfig
from src.optimization.ga  import GA,  GAConfig
from src.visualization.plots import (
    plot_convergence,
    plot_before_after,
    plot_signal_timing,
    save_figure,
)
from experiments.configs import SEEDS


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#F0F4F8"
PANEL   = "#FFFFFF"
ACCENT  = "#2563EB"
GA_COL  = "#DC2626"
TXT     = "#1E293B"
SUB     = "#64748B"
HDR_BG  = "#E8EDF5"   # collapsible section header background
HDR_HOV = "#D5E3F5"   # header hover background

LIVE_REFRESH_MS = 300


# ─────────────────────────────────────────────────────────────────────────────
# Module-level UI helpers (pure widget factories, no self dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _collapsible(parent, title, color=TXT, open_by_default=True):
    """
    Create a collapsible section with a clickable arrow-header.

    Returns the BODY frame where child widgets should be packed.
    The WRAPPER frame is always packed in `parent`, maintaining its position
    in the stack regardless of collapsed/expanded state.
    """
    state = [open_by_default]

    # Wrapper: always packed — fixes position in parent
    wrapper = tk.Frame(parent, bg=PANEL)
    wrapper.pack(fill="x", pady=(2, 0))

    # Header bar
    hdr = tk.Frame(wrapper, bg=HDR_BG, cursor="hand2")
    hdr.pack(fill="x")

    arrow = tk.Label(hdr, text="▼" if open_by_default else "▶",
                     bg=HDR_BG, fg=color, font=("Consolas", 9, "bold"))
    arrow.pack(side="left", padx=(8, 3), pady=4)
    tk.Label(hdr, text=title, bg=HDR_BG, fg=color,
             font=("Arial", 9, "bold")).pack(side="left", pady=4)

    # Body (content)
    body = tk.Frame(wrapper, bg=PANEL)
    if open_by_default:
        body.pack(fill="x", padx=4, pady=(2, 4))

    def _toggle(e=None):
        if state[0]:
            body.pack_forget()
            arrow.config(text="▶")
        else:
            body.pack(fill="x", padx=4, pady=(2, 4))
            arrow.config(text="▼")
        state[0] = not state[0]

    def _hover_on(e):
        hdr.config(bg=HDR_HOV)
        for w in hdr.winfo_children():
            try:
                w.config(bg=HDR_HOV)
            except tk.TclError:
                pass

    def _hover_off(e):
        hdr.config(bg=HDR_BG)
        for w in hdr.winfo_children():
            try:
                w.config(bg=HDR_BG)
            except tk.TclError:
                pass

    for w in [hdr, arrow] + list(hdr.winfo_children()):
        w.bind("<Button-1>", _toggle)
        w.bind("<Enter>",    _hover_on)
        w.bind("<Leave>",    _hover_off)

    return body


def _section_label(parent, text, color=TXT):
    tk.Label(parent, text=text, bg=PANEL,
             font=("Arial", 9, "bold"), fg=color
             ).pack(pady=(8, 2), padx=8, anchor="w")


def _radio_group(parent, var, pairs, padx=20):
    for text, val in pairs:
        ttk.Radiobutton(parent, text=text, variable=var,
                        value=val).pack(anchor="w", padx=padx, pady=1)


def _row(parent, name, var):
    f = tk.Frame(parent, bg=PANEL)
    f.pack(pady=2, padx=8, fill="x")
    tk.Label(f, text=name, bg=PANEL, anchor="w").pack(
        side="left", expand=True, fill="x")
    tk.Entry(f, textvariable=var, width=9).pack(side="right")


def _dropdown(parent, name, var, choices):
    f = tk.Frame(parent, bg=PANEL)
    f.pack(pady=2, padx=8, fill="x")
    tk.Label(f, text=name, bg=PANEL, anchor="w").pack(
        side="left", expand=True, fill="x")
    ttk.Combobox(f, textvariable=var, values=choices,
                 state="readonly", width=15).pack(side="right")


def _alg_divider(parent, text, color):
    """Thin coloured bar + bold label used to title PSO or GA blocks."""
    tk.Frame(parent, bg=color, height=2).pack(fill="x", padx=6, pady=(10, 0))
    tk.Label(parent, text=text, bg=PANEL, fg=color,
             font=("Arial", 9, "bold")).pack(anchor="w", padx=8, pady=(1, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class TrafficOptApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Traffic Signal Optimisation — PSO & GA")
        self.resizable(True, True)

        w = int(self.winfo_screenwidth()  * 0.92)
        h = int(self.winfo_screenheight() * 0.92)
        self.geometry(f"{w}x{h}")
        self.configure(bg=BG)

        # ── runtime state ────────────────────────────────────────────────────
        self.running          = False
        self.q                = queue.Queue()
        self.results          = []
        self._pso_results     = []
        self._ga_results      = []
        self.network_obj      = None
        self.baseline         = 0
        self._live_curves     = []
        self._live_curves_pso = []
        self._live_curves_ga  = []
        self._compare_mode    = False

        self._build_ui()
        self._poll()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        header = tk.Frame(self, bg=ACCENT, height=46)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Traffic Signal Timing Optimisation — PSO & GA",
            bg=ACCENT, fg="white", font=("Arial", 15, "bold"),
        ).pack(anchor="center", pady=12)

        body = tk.PanedWindow(self, orient="horizontal", bg=BG,
                              sashrelief="raised", sashwidth=5)
        body.pack(fill="both", expand=True)

        left  = tk.Frame(body, bg=PANEL)
        right = tk.Frame(body, bg=BG)
        body.add(left,  minsize=320, stretch="always")
        body.add(right, minsize=420, stretch="always")

        self._build_controls(left)
        self._build_tabs(right)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_controls(self, p):
        # Scrollable canvas so left panel never clips on small screens
        canvas = tk.Canvas(p, bg=PANEL, highlightthickness=0)
        scroll = ttk.Scrollbar(p, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=PANEL)
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Keep scroll region + inner width in sync with canvas
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            win, width=e.width))

        # Mousewheel scrolling
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ══════════════════════════════════════════════════════════════════════
        # [1] ALGORITHM SELECTOR  (always visible, no collapse)
        # ══════════════════════════════════════════════════════════════════════
        alg_hdr = tk.Frame(inner, bg=ACCENT)
        alg_hdr.pack(fill="x", pady=(0, 2))
        tk.Label(alg_hdr, text="  Algorithm", bg=ACCENT, fg="white",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=6)

        self.alg_var = tk.StringVar(value="pso")
        for text, val in [
            ("PSO only",          "pso"),
            ("GA only",           "ga"),
            ("Compare PSO vs GA", "compare"),
        ]:
            ttk.Radiobutton(inner, text=text, variable=self.alg_var,
                            value=val).pack(anchor="w", padx=18, pady=1)

        self.alg_var.trace_add("write", self._on_alg_change)

        shared_body = _collapsible(inner, "Shared run parameters", TXT)
        self.runs_var  = tk.StringVar(value="5")
        self.iters_var = tk.StringVar(value="200")
        _row(shared_body, "Number of runs",        self.runs_var)
        _row(shared_body, "Max iterations / gens", self.iters_var)

        # Declare shared vars here so both algorithm containers can reference them
        self.network_var = tk.StringVar(value="multi")
        self.init_var    = tk.StringVar(value="random")

        # ══════════════════════════════════════════════════════════════════════
        # [3] ALGORITHM-SPECIFIC SLOT
        #   A fixed container in `inner`.  PSO and GA sub-containers are packed
        #   inside it — so run-controls below always stay below, regardless of
        #   which sub-containers are shown or hidden.
        # ══════════════════════════════════════════════════════════════════════
        self._alg_slot = tk.Frame(inner, bg=PANEL)
        self._alg_slot.pack(fill="x")

        # ── PSO sub-container ─────────────────────────────────────────────────
        self._pso_container = tk.Frame(self._alg_slot, bg=PANEL)

        _alg_divider(self._pso_container, "PSO Parameters", ACCENT)

        # Network + Init appear first inside PSO block
        pso_net = _collapsible(self._pso_container, "Network", ACCENT)
        _radio_group(pso_net, self.network_var, [
            ("Single intersection",    "single"),
            ("Multi intersection (4)", "multi"),
        ])

        pso_init = _collapsible(self._pso_container,
                                "Initialisation strategy", ACCENT)
        _radio_group(pso_init, self.init_var, [
            ("Random",           "random"),
            ("Opposition-based", "opposition"),
        ])

        pso_top = _collapsible(self._pso_container,
                               "Parent selection — topology", ACCENT)
        self.topology_var = tk.StringVar(value="gbest")
        _radio_group(pso_top, self.topology_var, [
            ("Global best (gbest) — fast convergence", "gbest"),
            ("Local best  (lbest) — better diversity", "lbest"),
        ])

        pso_var = _collapsible(self._pso_container,
                               "Variation operators", ACCENT)
        self.velocity_var  = tk.StringVar(value="standard")
        self.crossover_var = tk.StringVar(value="none")
        _dropdown(pso_var, "Velocity strategy  (op A)", self.velocity_var,
                  ["standard", "constriction"])
        _dropdown(pso_var, "Crossover strategy (op B)", self.crossover_var,
                  ["none", "arithmetic"])

        pso_surv = _collapsible(self._pso_container,
                                "Survivor selection", ACCENT, open_by_default=False)
        self.pbest_var = tk.StringVar(value="greedy")
        self.temp_var  = tk.StringVar(value="0.05")
        _dropdown(pso_surv, "pbest strategy", self.pbest_var,
                  ["greedy", "probabilistic"])
        _row(pso_surv, "Temperature (prob only)", self.temp_var)

        pso_div = _collapsible(self._pso_container,
                               "Diversity — crowding restart", ACCENT,
                               open_by_default=False)
        self.diversity_var = tk.StringVar(value="0.0")
        _dropdown(pso_div, "Min dist from gbest", self.diversity_var,
                  ["0.0", "0.10", "0.20", "0.30"])

        pso_num = _collapsible(self._pso_container,
                               "PSO numerical parameters", ACCENT,
                               open_by_default=False)
        self.swarm_var = tk.StringVar(value="30")
        self.w_start   = tk.StringVar(value="0.9")
        self.w_end     = tk.StringVar(value="0.4")
        self.c1_var    = tk.StringVar(value="2.0")
        self.c2_var    = tk.StringVar(value="2.0")
        self.vmax_var  = tk.StringVar(value="0.2")
        for name, var in [
            ("Swarm size",         self.swarm_var),
            ("Inertia start w₀",   self.w_start),
            ("Inertia end  wₜ",    self.w_end),
            ("Cognitive c1",       self.c1_var),
            ("Social c2",          self.c2_var),
            ("Velocity max ratio", self.vmax_var),
        ]:
            _row(pso_num, name, var)

        self._pso_container.pack(fill="x")  # visible by default (PSO mode)

        # ── GA sub-container ──────────────────────────────────────────────────
        self._ga_container = tk.Frame(self._alg_slot, bg=PANEL)
        # NOT packed initially; shown by _on_alg_change when GA or compare

        _alg_divider(self._ga_container, "GA Parameters", GA_COL)

        # Network + Init appear first inside GA block (same vars as PSO = synced)
        ga_net = _collapsible(self._ga_container, "Network", GA_COL)
        _radio_group(ga_net, self.network_var, [
            ("Single intersection",    "single"),
            ("Multi intersection (4)", "multi"),
        ])

        ga_init = _collapsible(self._ga_container,
                               "Initialisation strategy", GA_COL)
        _radio_group(ga_init, self.init_var, [
            ("Random",           "random"),
            ("Opposition-based", "opposition"),
        ])

        ga_sel = _collapsible(self._ga_container,
                              "Parent selection", GA_COL)
        self.ga_selection_var  = tk.StringVar(value="tournament")
        self.ga_tournament_var = tk.StringVar(value="3")
        _radio_group(ga_sel, self.ga_selection_var, [
            ("Tournament (k-way)",       "tournament"),
            ("Roulette wheel (inv-fit)", "roulette"),
        ])
        _row(ga_sel, "Tournament k", self.ga_tournament_var)

        ga_mut = _collapsible(self._ga_container,
                              "Mutation operator  (op A)", GA_COL)
        self.ga_mutation_var  = tk.StringVar(value="uniform")
        self.ga_mrate_var     = tk.StringVar(value="0.15")
        self.ga_b_var         = tk.StringVar(value="2.0")
        _radio_group(ga_mut, self.ga_mutation_var, [
            ("Uniform  (gene-reset)",     "uniform"),
            ("Nonuniform  (Michalewicz)", "nonuniform"),
        ])
        _row(ga_mut, "Mutation rate (per gene)", self.ga_mrate_var)
        _row(ga_mut, "Nonuniform b  (shape)",    self.ga_b_var)

        ga_xov = _collapsible(self._ga_container,
                              "Crossover operator  (op B)", GA_COL)
        self.ga_crossover_var = tk.StringVar(value="arithmetic")
        self.ga_xrate_var     = tk.StringVar(value="0.90")
        self.ga_blend_var     = tk.StringVar(value="0.5")
        _radio_group(ga_xov, self.ga_crossover_var, [
            ("Whole Arithmetic", "arithmetic"),
            ("Blend BLX-α",      "blend"),
        ])
        _row(ga_xov, "Crossover rate", self.ga_xrate_var)
        _row(ga_xov, "BLX-α alpha",    self.ga_blend_var)

        ga_surv = _collapsible(self._ga_container,
                               "Survivor selection", GA_COL, open_by_default=False)
        self.ga_replacement_var = tk.StringVar(value="generational")
        self.ga_elite_var       = tk.StringVar(value="1")
        _radio_group(ga_surv, self.ga_replacement_var, [
            ("Generational μ→λ (+ elitism)", "generational"),
            ("Elitist (μ+λ) — pool & keep",  "elitist"),
        ])
        _row(ga_surv, "Elite count (generational)", self.ga_elite_var)

        ga_num = _collapsible(self._ga_container,
                              "Population & rates", GA_COL, open_by_default=False)
        self.ga_pop_var = tk.StringVar(value="50")
        _row(ga_num, "Population size", self.ga_pop_var)

        # ══════════════════════════════════════════════════════════════════════
        # [4] RUN CONTROLS  (always visible, always at bottom of inner)
        # ══════════════════════════════════════════════════════════════════════
        tk.Frame(inner, bg="#D1D9E6", height=1).pack(fill="x", padx=8,
                                                      pady=(10, 4))

        self._run_btn = tk.Button(
            inner,
            text="▶  Run PSO",
            bg=ACCENT, fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._run,
        )
        self._run_btn.pack(fill="x", padx=10, pady=(0, 4))

        self.progress = ttk.Progressbar(inner, maximum=100)
        self.progress.pack(fill="x", padx=10, pady=(0, 4))

        self.status_lbl = tk.Label(
            inner, text="Ready", bg=PANEL, fg=SUB,
            wraplength=290, justify="left",
            font=("Arial", 9),
        )
        self.status_lbl.pack(pady=(0, 2), padx=10, anchor="w")

        tk.Label(inner, text="Run log:", bg=PANEL, fg=SUB,
                 font=("Arial", 8)).pack(anchor="w", padx=10)
        self.log = tk.Text(inner, height=6, state="disabled",
                           wrap="word", font=("Consolas", 8),
                           relief="flat", bg="#F8FAFC", fg=TXT)
        self.log.pack(fill="x", padx=10, pady=(0, 10))

        # Set initial visibility and button text
        self._on_alg_change()

    # ── Algorithm-switch callback ─────────────────────────────────────────────

    def _on_alg_change(self, *_):
        alg = self.alg_var.get()

        # Hide both first so packing order is deterministic
        self._pso_container.pack_forget()
        self._ga_container.pack_forget()

        if alg in ("pso", "compare"):
            self._pso_container.pack(fill="x")
        if alg in ("ga", "compare"):
            self._ga_container.pack(fill="x")

        self._run_btn.config(text={
            "pso":     "▶  Run PSO",
            "ga":      "▶  Run GA",
            "compare": "▶  Compare PSO vs GA",
        }.get(alg, "▶  Run"))

    # ── Right panel (tabs) ────────────────────────────────────────────────────

    def _build_tabs(self, p):
        self.tabs = ttk.Notebook(p)
        self.tabs.pack(fill="both", expand=True)

        self.t_live    = tk.Frame(self.tabs, bg=BG)
        self.t_conv    = tk.Frame(self.tabs, bg=BG)
        self.t_before  = tk.Frame(self.tabs, bg=BG)
        self.t_timing  = tk.Frame(self.tabs, bg=BG)
        self.t_compare = tk.Frame(self.tabs, bg=BG)

        self.tabs.add(self.t_live,    text="Live convergence")
        self.tabs.add(self.t_conv,    text="All runs")
        self.tabs.add(self.t_before,  text="Before / After")
        self.tabs.add(self.t_timing,  text="Timing")
        self.tabs.add(self.t_compare, text="PSO vs GA")

        # Placeholder in compare tab
        tk.Label(
            self.t_compare,
            text="Run 'Compare PSO vs GA' to populate this tab.",
            bg=BG, fg=SUB, font=("Arial", 11),
        ).pack(expand=True)

        # Persistent live-convergence figure (constrained_layout avoids clipping)
        self._live_fig, self._live_ax = plt.subplots(
            figsize=(8, 5), constrained_layout=True)
        self._live_canvas = FigureCanvasTkAgg(self._live_fig, self.t_live)
        live_widget = self._live_canvas.get_tk_widget()
        live_widget.pack(fill="both", expand=True)

        # Bind resize to the CANVAS WIDGET so size is measured correctly
        def _resize_live(event):
            if event.width < 20 or event.height < 20:
                return
            dpi  = self._live_fig.dpi
            self._live_fig.set_size_inches(
                event.width / dpi, event.height / dpi, forward=False)
            self._live_canvas.draw_idle()

        live_widget.bind("<Configure>", _resize_live)

    # ─────────────────────────────────────────────────────────────────────────
    # Config builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pso_config(self):
        return PSOConfig(
            swarm_size         = int(self.swarm_var.get()),
            max_iterations     = int(self.iters_var.get()),
            w_start            = float(self.w_start.get()),
            w_end              = float(self.w_end.get()),
            c1                 = float(self.c1_var.get()),
            c2                 = float(self.c2_var.get()),
            v_max_ratio        = float(self.vmax_var.get()),
            topology           = self.topology_var.get(),
            init_strategy      = self.init_var.get(),
            velocity_strategy  = self.velocity_var.get(),
            crossover_strategy = self.crossover_var.get(),
            pbest_strategy     = self.pbest_var.get(),
            pbest_temperature  = float(self.temp_var.get()),
            diversity_min_dist = float(self.diversity_var.get()),
        )

    def _build_ga_config(self):
        return GAConfig(
            pop_size        = int(self.ga_pop_var.get()),
            max_generations = int(self.iters_var.get()),
            crossover_rate  = float(self.ga_xrate_var.get()),
            mutation_rate   = float(self.ga_mrate_var.get()),
            selection       = self.ga_selection_var.get(),
            tournament_size = int(self.ga_tournament_var.get()),
            mutation        = self.ga_mutation_var.get(),
            nonuniform_b    = float(self.ga_b_var.get()),
            crossover       = self.ga_crossover_var.get(),
            blend_alpha     = float(self.ga_blend_var.get()),
            replacement     = self.ga_replacement_var.get(),
            elite_count     = int(self.ga_elite_var.get()),
            init_strategy   = self.init_var.get(),
        )

    def _build_network(self):
        if self.network_var.get() == "single":
            return build_single_intersection()
        return build_multi_intersection()

    # ─────────────────────────────────────────────────────────────────────────
    # Run dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self):
        if self.running:
            return

        self.running          = True
        self.results          = []
        self._pso_results     = []
        self._ga_results      = []
        self._live_curves     = []
        self._live_curves_pso = []
        self._live_curves_ga  = []
        self.progress["value"] = 0

        alg   = self.alg_var.get()
        seeds = SEEDS[: int(self.runs_var.get())]
        self._compare_mode = (alg == "compare")

        if alg == "pso":
            cfg = self._build_pso_config()
            t   = threading.Thread(
                target=self._worker_pso, args=(cfg, seeds), daemon=True)
        elif alg == "ga":
            cfg = self._build_ga_config()
            t   = threading.Thread(
                target=self._worker_ga, args=(cfg, seeds), daemon=True)
        else:
            pso_cfg = self._build_pso_config()
            ga_cfg  = self._build_ga_config()
            t       = threading.Thread(
                target=self._worker_compare,
                args=(pso_cfg, ga_cfg, seeds), daemon=True)

        t.start()
        self._schedule_live_update()

    # ─────────────────────────────────────────────────────────────────────────
    # Worker threads
    # ─────────────────────────────────────────────────────────────────────────

    def _worker_pso(self, cfg, seeds):
        net = self._build_network()
        self.network_obj = net
        self.baseline    = net.baseline_fitness()
        lb, ub = net.bounds()
        pso    = PSO(net.evaluate, lb, ub, cfg)
        for i, s in enumerate(seeds):
            self.q.put(("status", f"[PSO] Run {i+1}/{len(seeds)}"))
            r = pso.run(s)
            self.results.append(r)
            self._live_curves.append(list(r.convergence))
            self.q.put(("progress", (i + 1) / len(seeds) * 100))
        self.q.put(("done", "pso"))

    def _worker_ga(self, cfg, seeds):
        net = self._build_network()
        self.network_obj = net
        self.baseline    = net.baseline_fitness()
        lb, ub = net.bounds()
        ga     = GA(net.evaluate, lb, ub, cfg)
        for i, s in enumerate(seeds):
            self.q.put(("status", f"[GA] Run {i+1}/{len(seeds)}"))
            r = ga.run(s)
            self.results.append(r)
            self._live_curves.append(list(r.convergence))
            self.q.put(("progress", (i + 1) / len(seeds) * 100))
        self.q.put(("done", "ga"))

    def _worker_compare(self, pso_cfg, ga_cfg, seeds):
        net = self._build_network()
        self.network_obj = net
        self.baseline    = net.baseline_fitness()
        lb, ub = net.bounds()
        n      = len(seeds)

        pso = PSO(net.evaluate, lb, ub, pso_cfg)
        for i, s in enumerate(seeds):
            self.q.put(("status", f"[PSO] Run {i+1}/{n}"))
            r = pso.run(s)
            self._pso_results.append(r)
            self._live_curves_pso.append(list(r.convergence))
            self.q.put(("progress", (i + 1) / (2 * n) * 100))

        ga = GA(net.evaluate, lb, ub, ga_cfg)
        for i, s in enumerate(seeds):
            self.q.put(("status", f"[GA]  Run {i+1}/{n}"))
            r = ga.run(s)
            self._ga_results.append(r)
            self._live_curves_ga.append(list(r.convergence))
            self.q.put(("progress", (n + i + 1) / (2 * n) * 100))

        self.q.put(("done", "compare"))

    # ─────────────────────────────────────────────────────────────────────────
    # Live convergence chart
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_live_update(self):
        self._update_live_chart()
        if self.running:
            self.after(LIVE_REFRESH_MS, self._schedule_live_update)

    def _update_live_chart(self):
        ax = self._live_ax
        ax.clear()

        if self._compare_mode:
            pso_curves = list(self._live_curves_pso)
            ga_curves  = list(self._live_curves_ga)
            if not pso_curves and not ga_curves:
                self._live_canvas.draw_idle()
                return

            for idx, c in enumerate(pso_curves):
                ax.plot(c, lw=0.9, alpha=0.45, color=ACCENT,
                        label="PSO" if idx == 0 else "_")
            if len(pso_curves) > 1:
                ml  = min(len(c) for c in pso_curves)
                arr = np.array([c[:ml] for c in pso_curves])
                ax.plot(arr.mean(0), lw=2.5, color=ACCENT, label="PSO mean")

            for idx, c in enumerate(ga_curves):
                ax.plot(c, lw=0.9, alpha=0.45, color=GA_COL,
                        ls="--", label="GA" if idx == 0 else "_")
            if len(ga_curves) > 1:
                ml  = min(len(c) for c in ga_curves)
                arr = np.array([c[:ml] for c in ga_curves])
                ax.plot(arr.mean(0), lw=2.5, color=GA_COL, ls="--",
                        label="GA mean")

            ax.set_title("Live convergence — PSO (blue) vs GA (red)")
        else:
            curves = list(self._live_curves)
            if not curves:
                self._live_canvas.draw_idle()
                return
            for idx, c in enumerate(curves):
                ax.plot(c, lw=1.2, alpha=0.6, label=f"Run {idx+1}")
            if len(curves) > 1:
                ml  = min(len(c) for c in curves)
                arr = np.array([c[:ml] for c in curves])
                ax.plot(arr.mean(0), lw=2.5, color="black",
                        label="Mean", zorder=5)
            ax.set_title("Live convergence (updates every run)")

        ax.set_xlabel("Iteration / Generation")
        ax.set_ylabel("Average Delay")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="upper right")
        self._live_canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Poll queue (main thread only)
    # ─────────────────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg, val = self.q.get_nowait()
                if msg == "status":
                    self.status_lbl.config(text=val)
                    self._log(val)
                elif msg == "progress":
                    self.progress["value"] = val
                elif msg == "done":
                    self._done(val)
        except Exception:
            pass
        self.after(100, self._poll)

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # Post-run visualisations
    # ─────────────────────────────────────────────────────────────────────────

    def _done(self, which):
        self.running = False
        self._update_live_chart()
        if which == "compare":
            self._finish_compare()
        else:
            self._finish_single(which.upper())

    def _finish_single(self, alg_name):
        fits = [r.best_fitness for r in self.results]
        best = min(self.results, key=lambda r: r.best_fitness)

        fig1 = plot_convergence([r.convergence for r in self.results],
                                title=f"{alg_name} — all runs convergence")
        fig2 = plot_before_after(self.baseline, fits)
        fig3 = plot_signal_timing(
            best.best_solution,
            self.network_obj.baseline_solution(),
            self.network_obj.n_intersections(),
            self.network_obj.intersections[0].n_phases,
        )

        self._show(fig1, self.t_conv)
        self._show(fig2, self.t_before)
        self._show(fig3, self.t_timing)
        self.tabs.select(self.t_conv)

        pct = (self.baseline - best.best_fitness) / self.baseline * 100
        self.status_lbl.config(
            text=f"{alg_name} done — best: {best.best_fitness:.3f} "
                 f"({pct:.1f}% vs baseline)")
        self._log(f"Baseline : {self.baseline:.3f}")
        self._log(f"Best     : {best.best_fitness:.3f}  ({pct:.1f}% improvement)")

    def _finish_compare(self):
        pso_fits = [r.best_fitness for r in self._pso_results]
        ga_fits  = [r.best_fitness for r in self._ga_results]
        pso_best = min(self._pso_results, key=lambda r: r.best_fitness)
        ga_best  = min(self._ga_results,  key=lambda r: r.best_fitness)

        fig1 = plot_convergence(
            [r.convergence for r in self._pso_results],
            title="PSO — all runs convergence")
        self._show(fig1, self.t_conv)

        fig2 = plot_before_after(self.baseline, pso_fits)
        self._show(fig2, self.t_before)

        fig3 = plot_signal_timing(
            pso_best.best_solution,
            self.network_obj.baseline_solution(),
            self.network_obj.n_intersections(),
            self.network_obj.intersections[0].n_phases,
        )
        self._show(fig3, self.t_timing)

        fig4 = self._make_comparison_fig(pso_fits, ga_fits)
        self._show(fig4, self.t_compare)
        self.tabs.select(self.t_compare)

        pso_pct = (self.baseline - pso_best.best_fitness) / self.baseline * 100
        ga_pct  = (self.baseline - ga_best.best_fitness)  / self.baseline * 100
        winner  = "PSO" if pso_best.best_fitness <= ga_best.best_fitness else "GA"
        self.status_lbl.config(
            text=f"PSO {pso_best.best_fitness:.3f} ({pso_pct:.1f}%)  |  "
                 f"GA {ga_best.best_fitness:.3f} ({ga_pct:.1f}%)  |  "
                 f"Winner: {winner}")
        self._log(f"Baseline : {self.baseline:.3f}")
        self._log(f"PSO best : {pso_best.best_fitness:.3f}  ({pso_pct:.1f}%)")
        self._log(f"GA  best : {ga_best.best_fitness:.3f}   ({ga_pct:.1f}%)")
        self._log(f"Winner   : {winner}")

    def _make_comparison_fig(self, pso_fits, ga_fits):
        pso_curves = [r.convergence for r in self._pso_results]
        ga_curves  = [r.convergence for r in self._ga_results]
        max_len    = max(len(c) for c in pso_curves + ga_curves)

        def _pad(c):
            return c + [c[-1]] * (max_len - len(c))

        pso_arr = np.array([_pad(c) for c in pso_curves])
        ga_arr  = np.array([_pad(c) for c in ga_curves])
        iters   = np.arange(max_len)

        fig, (ax_conv, ax_box) = plt.subplots(1, 2, figsize=(12, 5),
                                               constrained_layout=True)

        pso_mean, pso_std = pso_arr.mean(0), pso_arr.std(0)
        ga_mean,  ga_std  = ga_arr.mean(0),  ga_arr.std(0)

        ax_conv.plot(iters, pso_mean, lw=2, color=ACCENT, label="PSO mean")
        ax_conv.fill_between(iters, pso_mean - pso_std, pso_mean + pso_std,
                             alpha=0.18, color=ACCENT)
        ax_conv.plot(iters, ga_mean, lw=2, color=GA_COL, ls="--",
                     label="GA mean")
        ax_conv.fill_between(iters, ga_mean - ga_std, ga_mean + ga_std,
                             alpha=0.18, color=GA_COL)
        ax_conv.set_title("Convergence comparison")
        ax_conv.set_xlabel("Iteration / Generation")
        ax_conv.set_ylabel("Average Delay (lower is better)")
        ax_conv.legend(fontsize=9)

        bp = ax_box.boxplot(
            [pso_fits, ga_fits],
            labels=["PSO", "GA"],
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor(ACCENT); bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor(GA_COL); bp["boxes"][1].set_alpha(0.55)
        ax_box.set_title("Final best fitness — all runs")
        ax_box.set_ylabel("Best Fitness")

        fig.suptitle("PSO vs GA — head-to-head comparison",
                     fontsize=13, fontweight="bold")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Embed a matplotlib figure in a notebook tab and keep it resize-aware
    # ─────────────────────────────────────────────────────────────────────────

    def _show(self, fig, frame):
        # Remove old content
        for w in frame.winfo_children():
            w.destroy()

        canvas  = FigureCanvasTkAgg(fig, frame)
        widget  = canvas.get_tk_widget()

        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()

        # Toolbar at the bottom; canvas fills the rest.
        toolbar.pack(side="bottom", fill="x")
        widget.pack(fill="both", expand=True)

        # Force tkinter to compute actual pixel sizes BEFORE drawing so the
        # very first render is not clipped at the bottom or right.
        frame.update_idletasks()
        w_px = widget.winfo_width()
        h_px = widget.winfo_height()
        if w_px > 20 and h_px > 20:
            fig.set_size_inches(w_px / fig.dpi, h_px / fig.dpi, forward=False)

        canvas.draw()

        def _on_resize(event, _fig=fig, _canvas=canvas):
            if event.width < 20 or event.height < 20:
                return
            # Resize figure to match the canvas widget (toolbar excluded)
            _fig.set_size_inches(
                event.width / _fig.dpi, event.height / _fig.dpi, forward=False)
            # constrained_layout (set in plots.py) adjusts margins automatically;
            # no need to call tight_layout here.
            _canvas.draw_idle()

        widget.bind("<Configure>", _on_resize)
        plt.close(fig)


def launch():
    app = TrafficOptApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
