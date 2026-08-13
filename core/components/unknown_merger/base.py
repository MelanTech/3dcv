"""Unknown merger provider abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.types import Detection, Frame


class BaseUnknownMerger(ABC):
    """Produces and merges OCR candidate boxes for unknown competition classes."""

    @abstractmethod
    def infer(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        """Return boxes that should be sent to OCR as unknown-class candidates."""
        raise NotImplementedError

    def close(self) -> None:
        """Release optional resources."""
        return


class DisabledUnknownMerger(BaseUnknownMerger):
    """No-op provider used when unknown merging is disabled."""

    def infer(
        self,
        _frame: Frame,
        _detections: List[Detection],
        _table: int,
    ) -> List[Detection]:
        return []
