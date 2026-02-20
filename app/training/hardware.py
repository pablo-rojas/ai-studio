from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceOption:
    """Selectable training device metadata for the configuration UI."""

    id: str
    label: str
    type: str
    memory_total_mb: int | None = None

    def to_payload(self) -> dict[str, object]:
        """Serialize the option for templates and JSON payloads."""
        payload: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "type": self.type,
        }
        if self.memory_total_mb is not None:
            payload["memory_total_mb"] = self.memory_total_mb
        return payload


def list_device_options() -> list[DeviceOption]:
    """Return CPU + detected GPU devices for training selection."""
    options: list[DeviceOption] = [
        DeviceOption(
            id="cpu",
            label="CPU",
            type="cpu",
        )
    ]

    try:
        import torch
    except ImportError:
        return options

    if not torch.cuda.is_available():
        return options

    for index in range(torch.cuda.device_count()):
        label = f"GPU {index}"
        memory_total_mb: int | None = None
        try:
            device_name = torch.cuda.get_device_name(index)
            label = f"GPU {index} ({device_name})"
        except Exception:
            # Keep fallback label when the device name cannot be queried.
            pass

        try:
            properties = torch.cuda.get_device_properties(index)
            memory_total_mb = int(properties.total_memory / (1024 * 1024))
        except Exception:
            # Memory metadata is optional for UI rendering.
            memory_total_mb = None

        options.append(
            DeviceOption(
                id=f"gpu:{index}",
                label=label,
                type="gpu",
                memory_total_mb=memory_total_mb,
            )
        )
    return options
