"""Minimal Tk runtime panel for live recognition counts and graceful exit."""
from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Dict, Iterable, Optional


CountRow = tuple[tuple[str, ...], str]


class BaseCountGui(ABC):
    """Runtime count display with a cooperative exit request."""

    @abstractmethod
    def update(
        self,
        counts: Dict[str, int],
        *,
        elapsed_sec: Optional[float] = None,
        table: Optional[int] = None,
    ) -> bool:
        """Refresh runtime status and return whether a graceful exit was requested."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release GUI resources."""
        raise NotImplementedError


class NoopCountGui(BaseCountGui):
    """Disabled count panel."""

    def update(
        self,
        _counts: Dict[str, int],
        *,
        elapsed_sec: Optional[float] = None,
        table: Optional[int] = None,
    ) -> bool:
        return False

    def close(self) -> None:
        return


class TkCountGui(BaseCountGui):
    """Small Tk window containing one live count per result class."""

    def __init__(self, config: Dict, count_rows: Iterable[CountRow]):
        import tkinter as tk

        self._tk = tk
        self._closed = False
        self._exit_requested = False
        self._rows = tuple(
            (tuple(str(key) for key in keys), str(label))
            for keys, label in count_rows
        )
        self._fps_smoothing = min(
            0.99,
            max(0.0, float(config.get("fps_smoothing", 0.8))),
        )
        self._last_update_time = None
        self._fps = None
        self._root = tk.Tk()
        self._root.title(str(config.get("window_name", "3DCV Counts")))
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._request_exit)

        container = tk.Frame(self._root, padx=12, pady=10)
        container.grid(sticky="nsew")
        tk.Label(container, text="FPS", anchor="w", width=18).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self._fps_value = tk.StringVar(value="-")
        tk.Label(container, textvariable=self._fps_value, anchor="e", width=7).grid(
            row=0,
            column=1,
            sticky="e",
        )
        tk.Label(container, text="Time", anchor="w", width=18).grid(
            row=1,
            column=0,
            sticky="w",
        )
        self._elapsed_value = tk.StringVar(value="-")
        tk.Label(container, textvariable=self._elapsed_value, anchor="e", width=7).grid(
            row=1,
            column=1,
            sticky="e",
        )
        tk.Label(container, text="Table", anchor="w", width=18).grid(
            row=2,
            column=0,
            sticky="w",
        )
        self._table_value = tk.StringVar(value="-")
        tk.Label(container, textvariable=self._table_value, anchor="e", width=7).grid(
            row=2,
            column=1,
            sticky="e",
        )
        self._values = {}
        for row, (count_keys, display_name) in enumerate(self._rows, start=3):
            tk.Label(container, text=display_name, anchor="w", width=18).grid(
                row=row,
                column=0,
                sticky="w",
            )
            value = tk.StringVar(value="0")
            self._values[count_keys] = value
            tk.Label(container, textvariable=value, anchor="e", width=7).grid(
                row=row,
                column=1,
                sticky="e",
            )

        tk.Button(
            container,
            text="Exit",
            command=self._request_exit,
            width=10,
        ).grid(
            row=len(self._rows) + 3,
            column=0,
            columnspan=2,
            pady=(10, 0),
        )

    def update(
        self,
        counts: Dict[str, int],
        *,
        elapsed_sec: Optional[float] = None,
        table: Optional[int] = None,
    ) -> bool:
        if self._closed:
            return True
        self._update_fps()
        self._elapsed_value.set(self._format_elapsed(elapsed_sec))
        self._table_value.set("-" if table is None else str(int(table)))
        for count_keys, value in self._values.items():
            count = sum(int(counts.get(key, 0)) for key in count_keys)
            value.set(str(max(0, count)))
        try:
            self._pump_events()
        except self._tk.TclError:
            self._exit_requested = True
        return self._exit_requested

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass

    @staticmethod
    def _format_elapsed(elapsed_sec: Optional[float]) -> str:
        if elapsed_sec is None:
            return "-"
        elapsed = max(0.0, float(elapsed_sec))
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        tenths = int((elapsed - int(elapsed)) * 10)
        return f"{minutes:02d}:{seconds:02d}.{tenths}"

    def _request_exit(self) -> None:
        self._exit_requested = True

    def _pump_events(self) -> None:
        self._root.update_idletasks()
        self._root.update()

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_update_time is not None:
            elapsed = now - self._last_update_time
            if elapsed > 1e-6:
                instant_fps = 1.0 / elapsed
                if self._fps is None:
                    self._fps = instant_fps
                else:
                    alpha = self._fps_smoothing
                    self._fps = alpha * self._fps + (1.0 - alpha) * instant_fps
                self._fps_value.set(f"{self._fps:.1f}")
        self._last_update_time = now
