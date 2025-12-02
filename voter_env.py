import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# takes in a graph that is fixed, and prob per voter matrix. n.run compues asynptotically best result
# make voterend for graph and probs you want and 
class VoterEnv:
    def __init__(
        self,
        num_voters=5,
        num_alternatives=2,
        graph=None,
        p_correct=None,
        probs_per_voter=None,
        headless=True,
        record_data=False,
        seed=None
    ):
        self.num_voters = num_voters
        # Correct alternative is always alternative 0
        self.num_alternatives = num_alternatives

        # self.preferences[i] = j means voter i prefers alternative j
        self.preferences = None

        self.p_correct = p_correct  # Probability that each voter initially prefers the correct alternative

        if graph is None:
            print("No graph provided, using complete graph.")
            self.graph = np.ones((num_voters, num_voters)) - np.eye(num_voters)
        else:
            self.graph = graph

        # print(self.graph.shape, self.num_voters)

        # print(self.graph)

        # Visualization
        self.colors = ["green", "red", "blue", "yellow", "purple", "orange"]

        # Layout once so plots stay consistent frame-to-frame
        G = nx.from_numpy_array(self.graph)
        # self.pos = nx.spring_layout(G, seed=42)
        self.pos = nx.circular_layout(G)
        self.record_data = record_data

        # Headless config
        self.headless = headless
        self.out_dir = None
        if record_data:
            if self.headless:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.out_dir = os.path.join(".", "experiments", f"experiment{ts}")
                os.makedirs(self.out_dir, exist_ok=True)


        # Initialize state
        self.reset(probs_per_voter=probs_per_voter)

        # If headless, save reliability graph immediately at init
        if self.headless and record_data:
            self.plot_reliability(
                save_path=os.path.join(self.out_dir, "reliability.png")
            )
        # else:
            # Optional: still show reliability plot on init in interactive mode
            # self.plot_reliability()

    # probs_per_voter[i,j] = probability that voter i prefers alternative j
    def reset(self, probs_per_voter=None):
        assert (
            self.p_correct is not None or probs_per_voter is not None
        ), "Must provide correct probabilities per voter."

        self.probs_per_voter = probs_per_voter
        self.preferences = np.zeros((self.num_voters,), dtype=int)

        if probs_per_voter is not None:
            for i in range(self.num_voters):
                self.preferences[i] = np.random.choice(
                    self.num_alternatives, p=probs_per_voter[i]
                )
        else:
            for i in range(self.num_voters):
                self.preferences[i] = (
                    0
                    if np.random.rand() < self.p_correct
                    else np.random.choice(range(1, self.num_alternatives))
                )

    def _draw_preferences(self, ax=None, title_suffix=""):
        """Draw the preference graph onto provided axes (or the current axes)."""
        G = nx.from_numpy_array(self.graph)
        color_map = [self.colors[pref] for pref in self.preferences]
        if ax is None:
            ax = plt.gca()
        ax.clear()
        nx.draw(G, self.pos, node_color=color_map, with_labels=True, ax=ax)
        ax.set_title(f"Voter Preferences{title_suffix}")

    def plot_preferences(self, title_suffix="", save_path=None):
        """Plot (or save) the preferences graph with a clear title."""
        fig, ax = plt.subplots()
        self._draw_preferences(ax=ax, title_suffix=title_suffix)
        self._maybe_show_or_save(fig, save_path)
        plt.close(fig)

    def plot_reliability(self, save_path=None):
        """Heatmap-style node colors for probability of correct preference at reset, with a clear title."""
        G = nx.from_numpy_array(self.graph)
        reliability = np.zeros((self.num_voters,))

        for i in range(self.num_voters):
            if self.p_correct is not None:
                reliability[i] = self.p_correct
            elif self.probs_per_voter is not None:
                reliability[i] = self.probs_per_voter[i][0]
            else:
                reliability[i] = 0  # Default to 0 if no probabilities are provided

        color_map = [plt.cm.RdYlGn(r) for r in reliability]

        fig, ax = plt.subplots()
        nx.draw(G, self.pos, node_color=color_map, with_labels=True, ax=ax)
        ax.set_title("Voter Reliability (P(correct))")
        # Optional colorbar for clarity
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Probability correct")

        self._maybe_show_or_save(fig, save_path)
        plt.close(fig)

    def _maybe_show_or_save(self, fig, save_path):
        """Show fig if interactive; save if save_path provided or in headless mode."""
        if self.headless or save_path is not None:
            if save_path is None:
                # Default filename if none given
                save_path = os.path.join(self.out_dir or ".", "figure.png")
            fig.tight_layout()
            fig.savefig(save_path, dpi=200)
        else:
            plt.show()

    def step(self):
        # For each voter, find what the plurality of its neighbors prefer (-1 if tie)
        plurality_preferences = np.zeros((self.num_voters,), dtype=int)

        for i in range(self.num_voters):
            neighbor_indices = np.where(self.graph[i] == 1)[0]
            neighbor_preferences = self.preferences[neighbor_indices]

            counts = np.bincount(neighbor_preferences, minlength=self.num_alternatives)
            max_count = np.max(counts)
            if np.sum(counts == max_count) > 1:
                plurality_preferences[i] = -1  # tie
            else:
                plurality_preferences[i] = np.argmax(counts)

        # Update each voter's preference to match the plurality of its neighbors (if no tie), and keep the same if tie
        for i in range(self.num_voters):
            if plurality_preferences[i] != -1:
                self.preferences[i] = plurality_preferences[i]

    def run(self, iters, fps=2):
        """
        Run the environment for `iters` iterations.
        - Interactive mode: just steps; call plot_preferences() manually if you want visuals.
        - Headless mode: records an MP4 of the preferences over time to ./experiments/experiment[TIMESTAMP]/preferences.mp4
        """

        # Headless recording
        if self.record_data:
            if not self.headless:
                # Just simulate; user can call plot_preferences() whenever they like
                for _ in range(iters):
                    self.step()
                return
        
            video_path = os.path.join(self.out_dir, "preferences.mp4")
            fig, ax = plt.subplots()

            # Set up writer (requires ffmpeg to be available in the environment)
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps, metadata=dict(artist="VoterEnv"), bitrate=1800)

            with writer.saving(fig, video_path, dpi=200):
                # Initial frame (iteration 0)
                self._draw_preferences(ax=ax, title_suffix=" (iteration 0)")
                fig.tight_layout()
                writer.grab_frame()

                # Subsequent frames
                for t in range(1, iters + 1):
                    self.step()
                    self._draw_preferences(ax=ax, title_suffix=f" (iteration {t})")
                    fig.tight_layout()
                    writer.grab_frame()

            plt.close(fig)
        else:
            for _ in range(iters):
                self.step()


    def winner(self):
        counts = np.bincount(self.preferences, minlength=self.num_alternatives)
        max_count = np.max(counts)
        if np.sum(counts == max_count) > 1:
            return -1  # tie
        else:
            return np.argmax(counts)
