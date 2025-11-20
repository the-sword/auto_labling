import base64
import io
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import requests
from PIL import Image

import inference_config


# Mapping from canonical front-end labels (lowercased) to richer
# textual prompts used to query the SAM3 HTTP service. This allows the
# UI to keep a compact set of class names while SAM3 sees more
# descriptive phrases that are easier to understand.
PROMPT_GROUPS = {
    "obstacle": [
        "obstacle",
        "traffic cone",
        "traffic safety cone",
        "parking cone",
        "road cone",
        "traffic bollard",
        "road barrier",
    ],
    "dung": [
        "dung",
        "animal dung",
        "animal feces",
        "animal droppings",
        "poop on the ground",
    ],
    "fence": [
        "fence",
        "wire fence",
        "metal fence",
        "wooden fence",
    ],
    "adult": [
        "adult",
        "adult person",
        "man",
        "woman",
    ],
    "pet": [
        "pet",
        "dog",
        "cat",
        "pet dog",
        "pet cat",
    ],
    "leaf": [
        "leaf",
        "fallen leaf",
        "tree leaf",
    ],
    "charging station": [
        "charging station",
        "ev charging station",
        "ev charging pile",
    ],
    "manhole cover": [
        "manhole cover",
        "sewer cover",
    ],
    "water": [
        "water puddle",
        "water on the ground",
        "puddle",
    ],
    "flatstone": [
        "flat stone",
        "flagstone",
        "stone slab",
    ],
    "flat spray can": [
        "flat spray can",
        "spray can lying on the ground",
        "spray paint can",
    ],
    "pipeline": [
        "pipeline",
        "pipe on the ground",
    ],
    "mud": [
        "mud",
        "muddy ground",
    ],
    "child": [
        "child",
        "kid",
        "little boy",
        "little girl",
    ],
    "hedgehog": [
        "hedgehog",
    ],
    "fruilt": [
        "fruilt",
        "fruit",
        "fallen fruit",
    ],
    "green plants": [
        "green plants",
        "bush",
        "shrub",
    ],
    "grass": [
        "grass",
        "lawn",
    ],
    "road": [
        "road",
        "asphalt road",
        "pathway",
    ],
    "background": [
        "background",
    ],
}


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None


class Sam3HTTPInference:
    def __init__(self, endpoint: Optional[str] = None, timeout: float = 60.0) -> None:
        self.endpoint = endpoint or inference_config.SAM3_HTTP_URL
        self.timeout = timeout

    @staticmethod
    def _encode_image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _decode_mask_from_base64(mask_b64: str) -> Optional[np.ndarray]:
        if not mask_b64:
            return None
        try:
            if "," in mask_b64:
                mask_b64 = mask_b64.split(",", 1)[1]
            raw = base64.b64decode(mask_b64)
            img = Image.open(io.BytesIO(raw)).convert("L")
            arr = np.array(img)
            return (arr > 0).astype(np.uint8)
        except Exception:
            return None

    def _call_remote(self, image_b64: str, labels: List[str], threshold: float) -> List[dict]:
        if not self.endpoint:
            raise RuntimeError("SAM3 HTTP endpoint is not configured")

        payload = {
            "image": image_b64,
            "labels": labels,
            "threshold": float(threshold),
        }

        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", False):
            raise RuntimeError(str(data.get("error", "SAM3 HTTP inference failed")))

        detections = data.get("detections")
        if not isinstance(detections, list):
            return []
        return detections

    def segment(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[DetectionResult]:
        image_b64 = self._encode_image_to_base64(image)

        # Expand canonical labels into richer internal prompts.
        expanded_labels: List[str] = []
        label_alias = {}
        for lbl in labels:
            if not isinstance(lbl, str):
                continue
            canonical = lbl.strip()
            if not canonical:
                continue
            key = canonical.lower()
            prompts = PROMPT_GROUPS.get(key, [canonical])
            for p in prompts:
                if not isinstance(p, str):
                    continue
                p_clean = p.strip()
                if not p_clean:
                    continue
                expanded_labels.append(p_clean)
                # Map raw SAM3 prompt label back to canonical front-end label.
                label_alias[p_clean.lower()] = canonical

        # Fallback: if expansion somehow produced nothing, use the original
        # labels as prompts so the behavior stays reasonable.
        if not expanded_labels:
            expanded_labels = [str(l).strip() for l in labels if str(l).strip()]

        raw_dets = self._call_remote(image_b64, expanded_labels, threshold)

        results: List[DetectionResult] = []
        for det in raw_dets:
            try:
                score = float(det.get("score", 0.0))
            except Exception:
                score = 0.0
            if score < threshold:
                continue

            raw_label = str(det.get("label", "")).strip()
            # Use alias map to convert rich prompts back to canonical labels
            # defined by the front-end (e.g., "traffic cone" -> "obstacle").
            label = label_alias.get(raw_label.lower(), raw_label)
            box_dict = det.get("box") or {}
            try:
                xmin = int(box_dict.get("xmin", 0))
                ymin = int(box_dict.get("ymin", 0))
                xmax = int(box_dict.get("xmax", 0))
                ymax = int(box_dict.get("ymax", 0))
            except Exception:
                continue

            box = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            mask_b64 = det.get("mask")
            mask_arr = self._decode_mask_from_base64(mask_b64) if mask_b64 else None

            results.append(DetectionResult(score=score, label=label, box=box, mask=mask_arr))

        return results

    def segment_batch(self, images: List[Image.Image], labels: List[str], threshold: float = 0.3) -> List[List[DetectionResult]]:
        results: List[List[DetectionResult]] = []
        for image in images:
            per_image = self.segment(image, labels, threshold)
            results.append(per_image)
        return results
