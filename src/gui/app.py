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
from src.visualization.plots import (
    plot_convergence,
    plot_before_after,
    plot_signal_timing,
    save_figure,
)
from experiments.configs import SEEDS


BG     = "#F0F4F8"
PANEL  = "#FFFFFF"
ACCENT = "#2563EB"
TXT    = "#1E293B"
SUB    = "#64748B"

# How often (in ms) the live convergence chart refreshes while a run is active
LIVE_REFRESH_MS = 300


class TrafficPSOApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Traffic Signal Timing Optimisation — PSO")
        self.resizable(True, True)

        w = int(self.winfo_screenwidth()  * 0.90)
        h = int(self.winfo_screenheight() * 0.90)
        self.geometry(f"{w}x{h}")
        self.configure(bg=BG)

        self.running       = False
        self.q             = queue.Queue()
        self.results       = []
        self.network_obj   = None
        self.baseline      = 0
        self._live_curves  = []   # updated in real time from worker thread

        self._build_ui()
        self._poll()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):

        header = tk.Frame(self, bg=ACCENT, height=50)
        header.pack(fill="x")
        tk.Label(
            header,
            text="PSO-based Traffic Signal Optimisation",
            bg=ACCENT, fg="white",
            font=("Arial", 16, "bold"),
        ).pack(anchor="center", pady=10)

        # tk.PanedWindow (not ttk) supports minsize on each pane
        body = tk.PanedWindow(self, orient="horizontal", bg=BG,
                              sashrelief="raised", sashwidth=5)
        body.pack(fill="both", expand=True)

        left  = tk.Frame(body, bg=PANEL)
        right = tk.Frame(body, bg=BG)
        body.add(left,  minsize=310, stretch="always")
        body.add(right, minsize=400, stretch="always")

        self._build_controls(left)
        self._build_tabs(right)

    def _build_controls(self, p):

        # ── scroll wrapper so the panel never clips on small screens ─────────
        canvas = tk.Canvas(p, bg=PANEL, highlightthickness=0)
        scroll = ttk.Scrollbar(p, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=PANEL)
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            # Keep the inner frame exactly as wide as the canvas so widgets
            # stretch edge-to-edge and never overflow into the plot area.
            canvas.itemconfig(win, width=e.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _section(text):
            tk.Label(
                inner, text=text, bg=PANEL,
                font=("Arial", 11, "bold"), fg=TXT,
            ).pack(pady=(14, 4), padx=12, anchor="w")

        def _row(name, var):
            f = tk.Frame(inner, bg=PANEL)
            f.pack(pady=2, padx=12, fill="x")
            tk.Label(f, text=name, bg=PANEL, anchor="w").pack(side="left", expand=True, fill="x")
            tk.Entry(f, textvariable=var, width=9).pack(side="right")

        def _dropdown(name, var, choices):
            f = tk.Frame(inner, bg=PANEL)
            f.pack(pady=2, padx=12, fill="x")
            tk.Label(f, text=name, bg=PANEL, anchor="w").pack(side="left", expand=True, fill="x")
            ttk.Combobox(
                f, textvariable=var, values=choices,
                state="readonly", width=16,
            ).pack(side="right")

        # ── Network ──────────────────────────────────────────────────────────
        _section("Network")
        self.network_var = tk.StringVar(value="multi")
        for text, val in [("Single intersection", "single"),
                          ("Multi intersection (4)", "multi")]:
            ttk.Radiobutton(inner, text=text,
                            variable=self.network_var, value=val).pack(
                anchor="w", padx=24)

        # ── Parent selection ─────────────────────────────────────────────────
        _section("Parent selection — topology")
        self.topology_var = tk.StringVar(value="gbest")
        for text, val in [
            ("Global best  (gbest) — fast convergence", "gbest"),
            ("Local best   (lbest) — better diversity", "lbest"),
        ]:
            ttk.Radiobutton(inner, text=text,
                            variable=self.topology_var, value=val).pack(
                anchor="w", padx=24)

        # ── Initialisation ───────────────────────────────────────────────────
        _section("Initialisation strategy")
        self.init_var = tk.StringVar(value="random")
        for text, val in [("Random", "random"),
                          ("Opposition-based", "opposition")]:
            ttk.Radiobutton(inner, text=text,
                            variable=self.init_var, value=val).pack(
                anchor="w", padx=24)

        # ── Variation operator A: mutation ───────────────────────────────────
        _section("Variation operator A — velocity update (mutation)")
        self.velocity_var = tk.StringVar(value="standard")
        _dropdown(
            "Velocity strategy", self.velocity_var,
            ["standard", "constriction"],
        )

        # ── Variation operator B: crossover ──────────────────────────────────
        _section("Variation operator B — crossover (recombination)")
        self.crossover_var = tk.StringVar(value="none")
        _dropdown(
            "Crossover strategy", self.crossover_var,
            ["none", "arithmetic"],
        )

        # ── Survivor selection ───────────────────────────────────────────────
        _section("Survivor selection — pbest policy")
        self.pbest_var = tk.StringVar(value="greedy")
        _dropdown(
            "pbest strategy", self.pbest_var,
            ["greedy", "probabilistic"],
        )
        self.temp_var = tk.StringVar(value="0.05")
        _row("Temperature (prob only)", self.temp_var)

        # ── Diversity preservation ────────────────────────────────────────────
        _section("Diversity preservation — crowding")
        self.diversity_var = tk.StringVar(value="0.0")
        _dropdown(
            "Min dist from gbest", self.diversity_var,
            ["0.0", "0.10", "0.20", "0.30"],
        )

        # ── Numerical parameters ─────────────────────────────────────────────
        _section("Numerical parameters")
        self.swarm_var  = tk.StringVar(value="30")
        self.iters_var  = tk.StringVar(value="200")
        self.runs_var   = tk.StringVar(value="5")
        self.w_start    = tk.StringVar(value="0.9")
        self.w_end      = tk.StringVar(value="0.4")
        self.c1_var     = tk.StringVar(value="2.0")
        self.c2_var     = tk.StringVar(value="2.0")
        self.vmax_var   = tk.StringVar(value="0.2")

        for name, var in [
            ("Swarm size",          self.swarm_var),
            ("Max iterations",      self.iters_var),
            ("Number of runs",      self.runs_var),
            ("Inertia start",       self.w_start),
            ("Inertia end",         self.w_end),
            ("Cognitive c1",        self.c1_var),
            ("Social c2",           self.c2_var),
            ("Velocity max ratio",  self.vmax_var),
        ]:
            _row(name, var)

        # ── Run button ───────────────────────────────────────────────────────
        tk.Button(
            inner, text="▶  Run Optimisation",
            bg=ACCENT, fg="white",
            font=("Arial", 10, "bold"),
            command=self._run,
        ).pack(fill="x", padx=12, pady=(18, 4))

        self.progress = ttk.Progressbar(inner, maximum=100)
        self.progress.pack(fill="x", padx=12, pady=(0, 4))

        self.status_lbl = tk.Label(inner, text="Ready", bg=PANEL, fg=SUB)
        self.status_lbl.pack(pady=2)

        self.log = tk.Text(inner, height=8, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _build_tabs(self, p):

        self.tabs = ttk.Notebook(p)
        self.tabs.pack(fill="both", expand=True)

        self.t_live   = tk.Frame(self.tabs, bg=BG)
        self.t_conv   = tk.Frame(self.tabs, bg=BG)
        self.t_before = tk.Frame(self.tabs, bg=BG)
        self.t_timing = tk.Frame(self.tabs, bg=BG)

        self.tabs.add(self.t_live,   text="Live convergence")
        self.tabs.add(self.t_conv,   text="All runs")
        self.tabs.add(self.t_before, text="Before / After")
        self.tabs.add(self.t_timing, text="Timing")

        # Create a persistent live-convergence figure
        # Use constrained_layout so labels/titles are never clipped
        self._live_fig, self._live_ax = plt.subplots(
            figsize=(8, 5), constrained_layout=True
        )
        self._live_canvas = FigureCanvasTkAgg(self._live_fig, self.t_live)
        live_widget = self._live_canvas.get_tk_widget()
        live_widget.pack(fill="both", expand=True)

        # Resize the figure whenever the tab frame changes size
        def _resize_live(event):
            dpi = self._live_fig.dpi
            w_in = max(event.width  / dpi, 2)
            h_in = max(event.height / dpi, 2)
            self._live_fig.set_size_inches(w_in, h_in, forward=False)
            self._live_canvas.draw_idle()

        self.t_live.bind("<Configure>", _resize_live)

    # ─────────────────────────────────────────────────────────────────────────
    # Run logic
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self):

        if self.running:
            return

        self.running      = True
        self.results      = []
        self._live_curves = []
        self.progress["value"] = 0

        cfg = PSOConfig(
            swarm_size          = int(self.swarm_var.get()),
            max_iterations      = int(self.iters_var.get()),
            w_start             = float(self.w_start.get()),
            w_end               = float(self.w_end.get()),
            c1                  = float(self.c1_var.get()),
            c2                  = float(self.c2_var.get()),
            v_max_ratio         = float(self.vmax_var.get()),
            topology            = self.topology_var.get(),
            init_strategy       = self.init_var.get(),
            velocity_strategy   = self.velocity_var.get(),
            crossover_strategy  = self.crossover_var.get(),
            pbest_strategy      = self.pbest_var.get(),
            pbest_temperature   = float(self.temp_var.get()),
            diversity_min_dist  = float(self.diversity_var.get()),
        )

        seeds = SEEDS[: int(self.runs_var.get())]

        t = threading.Thread(target=self._worker, args=(cfg, seeds), daemon=True)
        t.start()

        self._schedule_live_update()

    def _worker(self, cfg, seeds):

        if self.network_var.get() == "single":
            net = build_single_intersection()
        else:
            net = build_multi_intersection()

        self.network_obj = net
        self.baseline    = net.baseline_fitness()

        lb, ub = net.bounds()
        pso    = PSO(net.evaluate, lb, ub, cfg)

        for i, s in enumerate(seeds):
            self.q.put(("status", f"Run {i + 1} / {len(seeds)}"))
            r = pso.run(s)
            self.results.append(r)
            # Share the new convergence curve with the live plot thread-safely
            self._live_curves.append(list(r.convergence))
            self.q.put(("progress", (i + 1) / len(seeds) * 100))

        self.q.put(("done", None))

    # ─────────────────────────────────────────────────────────────────────────
    # Live convergence chart
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_live_update(self):
        """Schedule periodic redraws of the live convergence tab."""
        self._update_live_chart()
        if self.running:
            self.after(LIVE_REFRESH_MS, self._schedule_live_update)

    def _update_live_chart(self):

        curves = list(self._live_curves)   # snapshot (avoids race)
        if not curves:
            return

        ax = self._live_ax
        ax.clear()

        for idx, c in enumerate(curves):
            ax.plot(c, lw=1.2, alpha=0.6,
                    label=f"Run {idx + 1}")

        # Show mean curve once we have more than one run
        if len(curves) > 1:
            min_len = min(len(c) for c in curves)
            arr     = np.array([c[:min_len] for c in curves])
            ax.plot(arr.mean(axis=0), lw=2.5, color="black",
                    label="Mean", zorder=5)

        ax.set_title("Live convergence (updates every run)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Delay")
        ax.legend(fontsize=8, loc="upper right")

        self._live_fig.tight_layout()
        self._live_canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Poll queue (main thread)
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
                    self._done()

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

    def _done(self):

        self.running = False
        self._update_live_chart()   # final refresh

        fits = [r.best_fitness for r in self.results]

        fig1 = plot_convergence([r.convergence for r in self.results],
                                title="All runs — convergence")
        fig2 = plot_before_after(self.baseline, fits)

        best = min(self.results, key=lambda r: r.best_fitness)
        fig3 = plot_signal_timing(
            best.best_solution,
            self.network_obj.baseline_solution(),
            self.network_obj.n_intersections(),
            self.network_obj.intersections[0].n_phases,
        )

        self._show(fig1, self.t_conv)
        self._show(fig2, self.t_before)
        self._show(fig3, self.t_timing)

        pct = (self.baseline - best.best_fitness) / self.baseline * 100
        self.status_lbl.config(
            text=f"Done — best delay: {best.best_fitness:.3f}  "
                 f"({pct:.1f}% improvement)"
        )
        self._log(f"Baseline: {self.baseline:.3f}")
        self._log(f"Best:     {best.best_fitness:.3f}  ({pct:.1f}% improvement)")

    def _show(self, fig, frame):

        for w in frame.winfo_children():
            w.destroy()

        # Apply constrained layout so nothing is clipped
        try:
            fig.set_constrained_layout(True)
        except Exception:
            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()

        # Resize figure to match the frame whenever the frame is resized
        def _on_resize(event, _fig=fig, _canvas=canvas):
            dpi  = _fig.dpi
            w_in = max(event.width  / dpi, 2)
            h_in = max(event.height / dpi, 2)
            _fig.set_size_inches(w_in, h_in, forward=False)
            _canvas.draw_idle()

        frame.bind("<Configure>", _on_resize)

        plt.close(fig)


def launch():
    app = TrafficPSOApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
