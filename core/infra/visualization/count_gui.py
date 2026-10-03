"""Minimal Tk runtime panel for live recognition counts and graceful exit."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

import cv2
import numpy as np

from core.types import Detection, Frame

CountRow = tuple[tuple[str, ...], str]
RgbColor = tuple[int, int, int]


def build_preview_rgb(
    frame: Frame,
    detections: Iterable[Detection],
    bbox_colors: dict[str, RgbColor],
    width: int = 480,
    height: int = 360,
) -> np.ndarray | None:
    """Build a small RGB final-stage preview without touching the source frame."""
    if frame.rgb is None:
        return None
    rgb = np.asarray(frame.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None

    source_height, source_width = rgb.shape[:2]
    if source_width <= 0 or source_height <= 0:
        return None
    preview_width = max(1, int(width))
    preview_height = max(1, int(height))
    scale = min(preview_width / source_width, preview_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
    offset_x = (preview_width - resized_width) // 2
    offset_y = (preview_height - resized_height) // 2
    preview[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized

    line_width = 2
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(preview_width - 1, offset_x + round(x1 * scale)))
        y1 = max(0, min(preview_height - 1, offset_y + round(y1 * scale)))
        x2 = max(0, min(preview_width - 1, offset_x + round(x2 * scale)))
        y2 = max(0, min(preview_height - 1, offset_y + round(y2 * scale)))
        if x2 <= x1 or y2 <= y1:
            continue
        color = bbox_colors.get(detection.class_name, (0, 255, 0))
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, line_width)
        label = f"{detection.class_name} {detection.score:.2f}"
        cv2.putText(
            preview,
            label,
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return np.ascontiguousarray(preview)


class BaseCountGui(ABC):
    """Runtime count display with a cooperative exit request."""

    def update_preview(
        self,
        _frame: Frame,
        _detections: Iterable[Detection],
        *,
        table: int | None = None,
    ) -> None:
        """Optionally refresh the final-stage image preview."""
        del table

    @abstractmethod
    def update(
        self,
        counts: dict[str, int],
        *,
        elapsed_sec: float | None = None,
        table: int | None = None,
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
        _counts: dict[str, int],
        *,
        elapsed_sec: float | None = None,
        table: int | None = None,
    ) -> bool:
        return False

    def close(self) -> None:
        return


class TkCountGui(BaseCountGui):
    """Tk runtime panel with final-stage preview and live class counts."""

    _PREVIEW_INTERVAL_SEC = 0.25
    _PREVIEW_WIDTH = 480
    _PREVIEW_HEIGHT = 360
    _MAX_EVENTS_PER_UPDATE = 8
    _BACKGROUND = "#1f232b"
    _SURFACE = "#2b303a"
    _TEXT = "#f4f7fb"
    _MUTED = "#b7beca"
    _BUTTON = "#343a46"
    _BUTTON_HOVER = "#414958"
    _BUTTON_ACTIVE = "#4c5668"
    _BUTTON_BORDER = "#586174"

    def __init__(
        self,
        config: dict,
        count_rows: Iterable[CountRow],
        bbox_colors: dict[str, RgbColor] | None = None,
    ):
        import tkinter as tk

        from PIL import Image, ImageTk

        self._tk = tk
        self._image_module = Image
        self._image_tk_module = ImageTk
        self._closed = False
        self._exit_requested = False
        self._preview_disabled = False
        self._bbox_colors = dict(bbox_colors or {})
        self._last_preview_time = None
        self._preview_photo = None
        self._rows = tuple(
            (tuple(str(key) for key in keys), str(label)) for keys, label in count_rows
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
        self._root.configure(background=self._BACKGROUND)
        self._root.protocol("WM_DELETE_WINDOW", self._request_exit)
        self._root.option_add("*Frame.background", self._BACKGROUND)
        self._root.option_add("*Label.background", self._BACKGROUND)
        self._root.option_add("*Label.foreground", self._TEXT)
        self._root.option_add("*Button.background", self._SURFACE)
        self._root.option_add("*Button.foreground", self._TEXT)

        container = tk.Frame(self._root, padx=12, pady=10)
        container.grid(sticky="nsew")

        preview_panel = tk.Frame(container)
        preview_panel.grid(row=0, column=0, sticky="n", padx=(0, 14))
        tk.Label(
            preview_panel,
            text="Final",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self._preview_photo = tk.PhotoImage(
            width=self._PREVIEW_WIDTH,
            height=self._PREVIEW_HEIGHT,
        )
        self._preview_photo.put(
            self._BACKGROUND,
            to=(0, 0, self._PREVIEW_WIDTH, self._PREVIEW_HEIGHT),
        )
        self._preview_label = tk.Label(
            preview_panel,
            image=self._preview_photo,
            anchor="center",
            relief="solid",
            borderwidth=1,
            background="#000000",
        )
        self._preview_label.grid(row=1, column=0, sticky="nw")

        counts_panel = tk.Frame(container)
        counts_panel.grid(row=0, column=1, sticky="n")
        tk.Label(counts_panel, text="FPS", anchor="w", width=18).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self._fps_value = tk.StringVar(value="-")
        tk.Label(counts_panel, textvariable=self._fps_value, anchor="e", width=7).grid(
            row=0,
            column=1,
            sticky="e",
        )
        tk.Label(counts_panel, text="Time", anchor="w", width=18).grid(
            row=1,
            column=0,
            sticky="w",
        )
        self._elapsed_value = tk.StringVar(value="-")
        tk.Label(
            counts_panel, textvariable=self._elapsed_value, anchor="e", width=7
        ).grid(
            row=1,
            column=1,
            sticky="e",
        )
        tk.Label(counts_panel, text="Table", anchor="w", width=18).grid(
            row=2,
            column=0,
            sticky="w",
        )
        self._table_value = tk.StringVar(value="-")
        tk.Label(
            counts_panel, textvariable=self._table_value, anchor="e", width=7
        ).grid(
            row=2,
            column=1,
            sticky="e",
        )
        self._values = {}
        for row, (count_keys, display_name) in enumerate(self._rows, start=3):
            tk.Label(counts_panel, text=display_name, anchor="w", width=18).grid(
                row=row,
                column=0,
                sticky="w",
            )
            value = tk.StringVar(value="0")
            self._values[count_keys] = value
            tk.Label(counts_panel, textvariable=value, anchor="e", width=7).grid(
                row=row,
                column=1,
                sticky="e",
            )

        self._exit_button = self._make_canvas_button(
            counts_panel,
            text="Exit",
            command=self._request_exit,
        )
        self._exit_button.grid(
            row=len(self._rows) + 3,
            column=0,
            columnspan=2,
            pady=(10, 0),
        )

    def update_preview(
        self,
        frame: Frame,
        detections: Iterable[Detection],
        *,
        table: int | None = None,
    ) -> None:
        """Refresh a throttled final-stage thumbnail without pumping GUI events."""
        del table
        if self._closed or self._preview_disabled:
            return
        now = time.perf_counter()
        if (
            self._last_preview_time is not None
            and now - self._last_preview_time < self._PREVIEW_INTERVAL_SEC
        ):
            return
        self._last_preview_time = now
        preview = build_preview_rgb(
            frame,
            detections,
            self._bbox_colors,
            width=self._PREVIEW_WIDTH,
            height=self._PREVIEW_HEIGHT,
        )
        if preview is None:
            return
        try:
            image = self._image_module.fromarray(preview)
            photo = self._image_tk_module.PhotoImage(image=image, master=self._root)
            self._preview_photo = photo
            self._preview_label.configure(image=photo, text="", width=0, height=0)
        except self._tk.TclError:
            self._exit_requested = True
        except (ValueError, TypeError, OSError, RuntimeError, cv2.error):
            self._preview_disabled = True

    def update(
        self,
        counts: dict[str, int],
        *,
        elapsed_sec: float | None = None,
        table: int | None = None,
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
    def _format_elapsed(elapsed_sec: float | None) -> str:
        if elapsed_sec is None:
            return "-"
        elapsed = max(0.0, float(elapsed_sec))
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        tenths = int((elapsed - int(elapsed)) * 10)
        return f"{minutes:02d}:{seconds:02d}.{tenths}"

    def _request_exit(self) -> None:
        self._exit_requested = True

    def _make_canvas_button(self, parent, *, text: str, command):
        button_width = 104
        button_height = 32
        canvas = self._tk.Canvas(
            parent,
            width=button_width,
            height=button_height,
            background=self._BACKGROUND,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        rect = canvas.create_rectangle(
            1,
            1,
            button_width - 1,
            button_height - 1,
            fill=self._BUTTON,
            outline=self._BUTTON_BORDER,
            width=1,
        )
        label = canvas.create_text(
            button_width // 2,
            button_height // 2,
            text=text,
            fill=self._TEXT,
            font=("TkDefaultFont", 11, "bold"),
        )

        def set_fill(color: str) -> None:
            canvas.itemconfigure(rect, fill=color)

        def is_inside(x: int, y: int) -> bool:
            return 0 <= x <= button_width and 0 <= y <= button_height

        def on_release(event) -> None:
            if is_inside(int(event.x), int(event.y)):
                set_fill(self._BUTTON_HOVER)
                command()
            else:
                set_fill(self._BUTTON)

        canvas.bind("<Enter>", lambda _event: set_fill(self._BUTTON_HOVER))
        canvas.bind("<Leave>", lambda _event: set_fill(self._BUTTON))
        canvas.bind("<ButtonPress-1>", lambda _event: set_fill(self._BUTTON_ACTIVE))
        canvas.bind("<ButtonRelease-1>", on_release)
        canvas.bind("<space>", lambda _event: command())
        canvas.bind("<Return>", lambda _event: command())
        for item in (rect, label):
            canvas.tag_bind(
                item, "<Enter>", lambda _event: set_fill(self._BUTTON_HOVER)
            )
            canvas.tag_bind(
                item, "<ButtonPress-1>", lambda _event: set_fill(self._BUTTON_ACTIVE)
            )
            canvas.tag_bind(item, "<ButtonRelease-1>", on_release)
        return canvas

    def _pump_events(self) -> None:
        flags = self._tk._tkinter.ALL_EVENTS | self._tk._tkinter.DONT_WAIT
        for _ in range(self._MAX_EVENTS_PER_UPDATE):
            if not self._root.tk.dooneevent(flags):
                break

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
