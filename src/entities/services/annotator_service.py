import numpy as np
import supervision as sv


class AnnotatorServiceBase:
    def __init__(self, anotator_name: str, hex_color: str, thickness: int = 1, text_thickness: int = 1, text_scale: float = 0.5):
        self.anotator_name = anotator_name

        self._validate(hex_color, thickness, text_scale)
        self.box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex(hex_color), thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(color=sv.Color.from_hex(hex_color), text_thickness=text_thickness, text_scale=text_scale)

    def _validate(self, hex_color: str, thickness: int, text_scale: float):
        if not hex_color.startswith("#") or len(hex_color) != 7:
            raise ValueError("Hex color must start with # and have 6 characters")

        if thickness <= 0 or text_scale <= 0:
            raise ValueError("Thickness and text scale must be greater than 0")

        if thickness > 100 or text_scale > 100:
            raise ValueError("Thickness and text scale must be less than 100")

    def annotate(self, annotated_frame: np.ndarray, detections, label: str):
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=[label])  # type: ignore

        return annotated_frame
