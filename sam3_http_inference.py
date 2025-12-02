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
    # 0 背景 background
    "background": [
        "background",
        "background area",
        "non-interest region in the image",
    ],

    # 1 障碍物 obstacle
    "obstacle": [
        "box",
        "trunk",
        "branch",
        "bush",
        "shrub",
        "rock",
        "stone",
        "outdoor furniture",
        "garden furniture",
        "table",
        "chair",
        "picnic blanket",
        "children toy",
        "toy car",
        "ball",
        "small animal on the lawn",
        "rabbit",
        "hamster",
        "irrigation equipment",
        "playground equipment",
        "slide",
        "swing",
        "statue",
        "trash bin",
        "mailbox",
        "pole",
        "construction sign",
        "garden tool on the ground",
    ],

    # 2 粪便 stool  （兼容旧标签 dung）
    "stool": [
        "pet stool",
        "pet feces",
        "animal stool",
        "animal feces",
        "animal droppings",
        "poop on the grass",
        "poop on the ground",
    ],
    "dung": [  # backward-compatible alias
        "pet stool",
        "animal dung",
        "animal feces",
        "animal droppings",
        "poop on the ground",
    ],

    # 3 平式喷罐头 flatspraycan  （兼容旧标签 flat spray can）
    "flatspraycan": [
        "flat spray can",
        "spray can lying on the ground",
        "spray paint can lying flat",
    ],
    "flat spray can": [
        "flat spray can",
        "spray can lying on the ground",
        "spray paint can lying flat",
    ],

    # 4 管线 pipeline
    "pipeline": [
        "pipeline",
        "cable on the ground",
        "electric cable",
        "water pipe",
        "garden hose",
        "hose on the ground",
        "rope on the ground",
    ],

    # 5 栅栏 fence
    "fence": [
        "fence",
        "wooden fence",
        "metal fence",
        "wire fence",
        "garden fence",
        "yard fence",
        "barrier fence",
    ],

    # 6 泥土 mud
    "mud": [
        "mud",
        "muddy ground",
        "bare soil",
        "exposed soil",
    ],

    # 7 成人 adult
    "adult": [
        "adult",
        "adult person",
        "man",
        "woman",
    ],

    # 8 儿童 child
    "child": [
        "child",
        "kid",
        "little boy",
        "little girl",
    ],

    # 9 宠物 pet
    "pet": [
        "pet",
        "dog",
        "cat",
        "pet dog",
        "pet cat",
    ],

    # 10 刺猬 hedgehog
    "hedgehog": [
        "hedgehog",
    ],

    # 11 落叶 leaf
    "leaf": [
        "leaf",
        "fallen leaf",
        "fallen leaves",
        "pile of leaves",
    ],

    # 12 果实 fruit
    "fruit": [
        "fruit",
        "fallen fruit",
        "fallen apple",
        "fallen apples",
        "fallen pear",
        "pine cone",
    ],

    # 13 充电桩 chargingstation  （兼容 charging station）
    "chargingstation": [
        "charging station",
        "robot mower charging station",
        "ev charging station",
        "ev charging pile",
    ],
    "charging station": [
        "charging station",
        "robot mower charging station",
        "ev charging station",
        "ev charging pile",
    ],

    # 14 类草绿植 greenplants  （兼容 green plants）
    "greenplants": [
        "green plants",
        "ornamental grass",
        "low green plants",
        "low shrubs",
        "flower bed",
    ],
    "green plants": [
        "green plants",
        "ornamental grass",
        "low green plants",
        "low shrubs",
        "flower bed",
    ],

    # 15 井盖 manholecover  （兼容 manhole cover）
    "manholecover": [
        "manhole cover",
        "round manhole cover",
        "square manhole cover",
        "sewer cover",
    ],
    "manhole cover": [
        "manhole cover",
        "round manhole cover",
        "square manhole cover",
        "sewer cover",
    ],

    # 16 草 grass
    "grass": [
        "grass",
        "lawn",
        "grass lawn",
        "grass field",
    ],

    # 17 水面 water
    "water": [
        "water puddle",
        "puddle",
        "water on the ground",
        "water surface",
        "pond water surface",
        "swimming pool water surface",
    ],

    # 18 路面 road
    "road": [
        "road surface",
        "asphalt road",
        "concrete road",
        "paved road",
        "paved path",
        "sidewalk",
    ],

    # 19 石板 flatstone
    "flatstone": [
        "flat stone",
        "stone slab",
        "flagstone",
        "paving stone",
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
