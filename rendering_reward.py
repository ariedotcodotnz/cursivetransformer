"""Rendering-aware rewards for autoregressive handwriting generation.

The model emits discrete stroke tokens, but the quality we care about is the
rendered handwriting.  This module closes that loop without importing the model
or dataset code.  It deliberately fits the raster canvas from the *target only*;
predictions are rendered with the same transform and therefore cannot obtain a
better score by changing their own bounds.

Coordinates are Cartesian ``(x, y, pen_down)`` arrays.  A single array denotes
one word, while a sequence of arrays denotes words laid out left to right.  Use
``polar_words_to_cartesian`` for the model's decoded ``(r, theta, pen_down)``
offsets.  Token-level syntax can be checked independently with
``analyze_token_sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


_EPS = 1e-8


@dataclass(frozen=True)
class RenderConfig:
    """Rasterization and anti-reward-hacking settings."""

    width: int = 384
    height: int = 128
    padding_fraction: float = 0.08
    line_width_px: float = 2.0
    blur_sigmas: tuple[float, ...] = (0.75, 2.0, 5.0)
    distance_clip_px: float = 18.0
    word_gap: float = 0.18
    minimum_target_extent: float = 0.25
    max_abs_coordinate: float = 1e5
    min_ink_pixels: int = 3
    minimum_ink_mass_ratio: float = 0.05
    length_band: tuple[float, float] = (0.75, 1.33)
    length_limits: tuple[float, float] = (0.5, 2.0)
    hard_oob_fraction: float = 0.95
    hard_failure_ceiling: float = -0.25

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 8:
            raise ValueError("canvas dimensions must both be at least 8 pixels")
        if not 0.0 <= self.padding_fraction < 0.45:
            raise ValueError("padding_fraction must be in [0, 0.45)")
        if self.line_width_px <= 0.0:
            raise ValueError("line_width_px must be positive")
        if not self.blur_sigmas or any(sigma < 0.0 for sigma in self.blur_sigmas):
            raise ValueError("blur_sigmas must contain non-negative values")
        low, high = self.length_band
        min_ratio, max_ratio = self.length_limits
        if not 0.0 < min_ratio < low <= 1.0 <= high < max_ratio:
            raise ValueError("length limits must surround a band containing 1.0")
        if self.hard_failure_ceiling > 0.0:
            raise ValueError("hard_failure_ceiling must not be positive")
        if not 0.0 <= self.minimum_ink_mass_ratio < 1.0:
            raise ValueError("minimum_ink_mass_ratio must be in [0, 1)")


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the interpretable reward components.

    Penalties have values in ``[-1, 0]`` and positive scores in ``[-1, 1]``.
    The final reward is an unnormalized weighted sum, matching common online RL
    reward aggregation and keeping each term easy to inspect in W&B.
    """

    shape: float = 1.0
    style: float = 0.25
    word_count: float = 0.20
    ended: float = 0.15
    validity: float = 0.20
    out_of_bounds: float = 0.35
    length: float = 0.10
    blank: float = 0.75
    hard_failure: float = 1.0


@dataclass(frozen=True)
class TokenSpec:
    """Token ranges needed to validate the alternating theta/radius grammar.

    Ranges are half-open.  ``TokenSpec.from_dataset(dataset)`` supports the
    current ``StrokeDataset`` without importing :mod:`data`.
    """

    theta_start: int
    theta_stop: int
    radius_start: int
    radius_stop: int
    word_token: int
    end_token: int
    pad_token: int
    bos_token: int | None = None

    @classmethod
    def from_dataset(cls, dataset: Any) -> "TokenSpec":
        radius_start = int(dataset.cumulative_sizes[0])
        theta_start = int(dataset.cumulative_sizes[1])
        bos_token = getattr(dataset, "BOS_TOKEN", None)
        return cls(
            theta_start=theta_start,
            theta_stop=theta_start + len(dataset.theta_bins),
            radius_start=radius_start,
            radius_stop=radius_start + len(dataset.r_bins),
            word_token=int(dataset.WORD_TOKEN),
            end_token=int(dataset.END_TOKEN),
            pad_token=int(dataset.PAD_TOKEN),
            bos_token=int(bos_token) if bos_token is not None else None,
        )


@dataclass(frozen=True)
class SequenceDiagnostics:
    """Generation metadata consumed by ``compute_rendering_reward``."""

    ended: bool | None = None
    valid: bool = True
    word_count: int | None = None
    token_length: int | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanvasTransform:
    """An immutable world-to-pixel transform fitted from target ink only."""

    scale: float
    x_offset: float
    y_offset: float
    width: int
    height: int
    target_bounds: tuple[float, float, float, float]

    def apply(self, xy: np.ndarray) -> np.ndarray:
        values = np.asarray(xy, dtype=np.float64)
        out = values.copy()
        out[..., 0] = values[..., 0] * self.scale + self.x_offset
        out[..., 1] = values[..., 1] * self.scale + self.y_offset
        return out


@dataclass
class RenderedPair:
    """Fixed-canvas target and prediction rasters plus rendering diagnostics."""

    prediction: np.ndarray
    target: np.ndarray
    transform: CanvasTransform
    out_of_bounds_fraction: float
    prediction_valid: bool
    target_valid: bool
    prediction_reasons: tuple[str, ...] = ()
    target_reasons: tuple[str, ...] = ()
    prediction_words: list[np.ndarray] = field(default_factory=list, repr=False)
    target_words: list[np.ndarray] = field(default_factory=list, repr=False)


@dataclass
class RewardResult:
    """Rendering reward with W&B-friendly scalar components."""

    total: float
    components: dict[str, float]
    diagnostics: dict[str, Any]

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        result = {f"{prefix}total": float(self.total)}
        result.update(
            {f"{prefix}{name}": float(value) for name, value in self.components.items()}
        )
        for name, value in self.diagnostics.items():
            if isinstance(value, (bool, int, float, np.number)):
                result[f"{prefix}{name}"] = float(value)
        return result


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def analyze_token_sequence(tokens: Any, spec: TokenSpec) -> SequenceDiagnostics:
    """Validate a generated sequence against the theta/radius word grammar.

    The function is intentionally strict: stroke tokens must alternate theta then
    radius, word separators must occur as exactly two adjacent ``WORD`` tokens,
    and everything after ``END`` must be padding.  Missing ``END`` is reported as
    truncation through ``ended=False`` but is kept separate from syntax validity.
    """

    reasons: list[str] = []
    try:
        raw = _as_numpy(tokens)
    except Exception:
        return SequenceDiagnostics(False, False, 0, 0, ("unreadable_tokens",))

    if raw.ndim != 1:
        reasons.append("tokens_not_1d")
        raw = raw.reshape(-1)
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            numeric = raw.astype(np.float64)
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                return SequenceDiagnostics(False, False, 0, 0, ("non_integer_tokens",))
            raw = numeric.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            return SequenceDiagnostics(False, False, 0, 0, ("non_integer_tokens",))
    else:
        raw = raw.astype(np.int64, copy=False)

    if spec.bos_token is not None and len(raw) and raw[0] == spec.bos_token:
        raw = raw[1:]

    end_positions = np.flatnonzero(raw == spec.end_token)
    ended = bool(len(end_positions))
    if ended:
        end_index = int(end_positions[0])
        content = raw[:end_index]
        if np.any(raw[end_index + 1 :] != spec.pad_token):
            reasons.append("non_padding_after_end")
        token_length = end_index + 1
    else:
        nonpad = np.flatnonzero(raw != spec.pad_token)
        content_stop = int(nonpad[-1] + 1) if len(nonpad) else 0
        content = raw[:content_stop]
        token_length = content_stop

    if np.any(content == spec.pad_token):
        reasons.append("padding_before_end")

    word_count = 0
    points_in_word = 0
    index = 0
    grammar_valid = True
    while index < len(content):
        token = int(content[index])
        if spec.theta_start <= token < spec.theta_stop:
            if index + 1 >= len(content):
                reasons.append("unpaired_theta")
                grammar_valid = False
                break
            radius = int(content[index + 1])
            if not spec.radius_start <= radius < spec.radius_stop:
                reasons.append("theta_not_followed_by_radius")
                grammar_valid = False
                index += 1
                continue
            points_in_word += 1
            index += 2
            continue

        if token == spec.word_token:
            paired = index + 1 < len(content) and content[index + 1] == spec.word_token
            if not paired:
                reasons.append("unpaired_word_token")
                grammar_valid = False
                index += 1
                continue
            if points_in_word == 0:
                reasons.append("empty_word")
                grammar_valid = False
            else:
                word_count += 1
            points_in_word = 0
            index += 2
            continue

        if spec.radius_start <= token < spec.radius_stop:
            reasons.append("radius_without_theta")
        else:
            reasons.append("invalid_token")
        grammar_valid = False
        index += 1

    if points_in_word:
        word_count += 1
    elif len(content) == 0:
        reasons.append("empty_sequence")
        grammar_valid = False
    elif content[-1] == spec.word_token:
        reasons.append("trailing_word_boundary")
        grammar_valid = False

    if not ended:
        reasons.append("missing_end")
    syntax_reasons = {
        "tokens_not_1d",
        "non_padding_after_end",
        "padding_before_end",
        "unpaired_theta",
        "theta_not_followed_by_radius",
        "unpaired_word_token",
        "empty_word",
        "radius_without_theta",
        "invalid_token",
        "empty_sequence",
        "trailing_word_boundary",
    }
    valid = grammar_valid and not any(reason in syntax_reasons for reason in reasons)
    return SequenceDiagnostics(ended, valid, word_count, token_length, tuple(dict.fromkeys(reasons)))


def polar_offsets_to_cartesian(offsets: Any) -> np.ndarray:
    """Convert one decoded ``(radius, theta, pen)`` word to absolute points."""

    values = _as_numpy(offsets).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("polar offsets must have shape (N, 3)")
    dx = values[:, 0] * np.cos(values[:, 1])
    dy = values[:, 0] * np.sin(values[:, 1])
    xy = np.cumsum(np.column_stack((dx, dy)), axis=0)
    return np.column_stack((xy, values[:, 2]))


def polar_words_to_cartesian(words: Any) -> list[np.ndarray]:
    """Convert a decoded sequence of polar-offset words to Cartesian words."""

    if isinstance(words, np.ndarray) and words.ndim == 2:
        return [polar_offsets_to_cartesian(words)]
    return [polar_offsets_to_cartesian(word) for word in words]


def _coerce_words(
    strokes: Any, max_abs_coordinate: float
) -> tuple[list[np.ndarray], bool, tuple[str, ...]]:
    reasons: list[str] = []
    if strokes is None:
        return [], True, ()

    try:
        direct: np.ndarray | None = _as_numpy(strokes)
    except Exception:
        direct = None

    if direct is not None and direct.ndim == 2 and (direct.shape[1] if direct.shape else 0) in (2, 3):
        candidates: Sequence[Any] = [direct]
    elif direct is not None and direct.ndim == 3 and direct.shape[2] in (2, 3):
        candidates = list(direct)
    elif direct is not None and direct.size == 0:
        candidates = []
    else:
        try:
            candidates = list(strokes)
        except TypeError:
            return [], False, ("strokes_not_iterable",)

    words: list[np.ndarray] = []
    valid = True
    for word_index, candidate in enumerate(candidates):
        try:
            word = _as_numpy(candidate).astype(np.float64, copy=True)
        except (TypeError, ValueError, OverflowError):
            valid = False
            reasons.append(f"word_{word_index}_not_numeric")
            continue
        if word.ndim != 2 or word.shape[1] not in (2, 3):
            valid = False
            reasons.append(f"word_{word_index}_bad_shape")
            continue
        if word.shape[1] == 2:
            word = np.column_stack((word, np.ones(len(word), dtype=np.float64)))
        if len(word) == 0:
            words.append(word.reshape(0, 3))
            continue

        finite_xy = np.isfinite(word[:, :2]).all(axis=1)
        bounded_xy = np.abs(word[:, :2]).max(axis=1) <= max_abs_coordinate
        finite_pen = np.isfinite(word[:, 2])
        good = finite_xy & bounded_xy & finite_pen
        if not good.all():
            valid = False
            reasons.append(f"word_{word_index}_nonfinite_or_extreme")
            word[~good, :2] = 0.0
            word[~good, 2] = 0.0
        finite_pen_values = word[finite_pen, 2]
        if len(finite_pen_values) and not np.isin(finite_pen_values, (0.0, 1.0)).all():
            valid = False
            reasons.append(f"word_{word_index}_invalid_pen_state")
        word[:, 2] = (word[:, 2] > 0.5).astype(np.float64)
        words.append(word[:, :3])

    return words, valid, tuple(dict.fromkeys(reasons))


def layout_words(words: Sequence[np.ndarray], word_gap: float = 0.18) -> list[np.ndarray]:
    """Lay out local-coordinate words without changing their offsets or shape.

    In particular, this does not normalize a prediction by its own minimum x.
    A generated large initial jump must remain visible to the fixed-canvas OOB
    check instead of being silently translated back onto the target.
    """

    result: list[np.ndarray] = []
    cursor = 0.0
    for word in words:
        placed = np.asarray(word, dtype=np.float64).copy()
        ink = placed[:, 2] > 0.5 if len(placed) else np.zeros(0, dtype=bool)
        if ink.any():
            placed[:, 0] += cursor
            cursor = float(np.max(placed[ink, 0])) + word_gap
        result.append(placed)
    return result


def _ink_runs(words: Sequence[np.ndarray]):
    for word in words:
        start: int | None = None
        for index, point in enumerate(word):
            if point[2] > 0.5:
                if start is None:
                    start = index
            elif start is not None:
                yield word[start:index, :2]
                start = None
        if start is not None:
            yield word[start:, :2]


def _ink_word_count(words: Sequence[np.ndarray]) -> int:
    return sum(bool(len(word) and np.any(word[:, 2] > 0.5)) for word in words)


def _ink_points(words: Sequence[np.ndarray]) -> np.ndarray:
    points = [word[word[:, 2] > 0.5, :2] for word in words if len(word)]
    points = [point for point in points if len(point)]
    return np.concatenate(points, axis=0) if points else np.empty((0, 2), dtype=np.float64)


def _fit_target_transform(words: Sequence[np.ndarray], config: RenderConfig) -> CanvasTransform:
    points = _ink_points(words)
    if len(points):
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
    else:
        x_min = y_min = -0.5
        x_max = y_max = 0.5

    span_x = max(float(x_max - x_min), config.minimum_target_extent)
    span_y = max(float(y_max - y_min), config.minimum_target_extent)
    center_x = float(x_min + x_max) / 2.0
    center_y = float(y_min + y_max) / 2.0
    x_min, x_max = center_x - span_x / 2.0, center_x + span_x / 2.0
    y_min, y_max = center_y - span_y / 2.0, center_y + span_y / 2.0

    pad = config.padding_fraction * min(config.width, config.height)
    inner_width = max(config.width - 1.0 - 2.0 * pad, 1.0)
    inner_height = max(config.height - 1.0 - 2.0 * pad, 1.0)
    scale = min(inner_width / span_x, inner_height / span_y)
    x_offset = pad + (inner_width - span_x * scale) / 2.0 - x_min * scale
    y_offset = pad + (inner_height - span_y * scale) / 2.0 - y_min * scale
    return CanvasTransform(
        scale=float(scale),
        x_offset=float(x_offset),
        y_offset=float(y_offset),
        width=config.width,
        height=config.height,
        target_bounds=(float(x_min), float(y_min), float(x_max), float(y_max)),
    )


def _clip_segment(
    start: np.ndarray, end: np.ndarray, width: int, height: int, margin: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Liang-Barsky clip returning endpoints and the retained length fraction."""

    x0, y0 = map(float, start)
    x1, y1 = map(float, end)
    dx, dy = x1 - x0, y1 - y0
    bounds = (-margin, width - 1.0 + margin, -margin, height - 1.0 + margin)
    p = (-dx, dx, -dy, dy)
    q = (x0 - bounds[0], bounds[1] - x0, y0 - bounds[2], bounds[3] - y0)
    low, high = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < _EPS:
            if qi < 0.0:
                return None
            continue
        ratio = qi / pi
        if pi < 0.0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return None
    clipped_start = np.array((x0 + low * dx, y0 + low * dy))
    clipped_end = np.array((x0 + high * dx, y0 + high * dy))
    return clipped_start, clipped_end, max(0.0, high - low)


def _stamp(image: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float) -> None:
    center_x = np.rint(x).astype(np.int64)
    center_y = np.rint(y).astype(np.int64)
    integer_radius = max(1, int(np.ceil(radius)))
    for dy in range(-integer_radius, integer_radius + 1):
        for dx in range(-integer_radius, integer_radius + 1):
            if dx * dx + dy * dy > radius * radius + 0.5:
                continue
            px = center_x + dx
            py = center_y + dy
            inside = (px >= 0) & (px < image.shape[1]) & (py >= 0) & (py < image.shape[0])
            image[py[inside], px[inside]] = 1.0


def rasterize_strokes(
    words: Sequence[np.ndarray], transform: CanvasTransform, line_width_px: float = 2.0
) -> np.ndarray:
    """Rasterize Cartesian words using an already fixed canvas transform."""

    image = np.zeros((transform.height, transform.width), dtype=np.float32)
    radius = max(line_width_px / 2.0, 0.5)
    for run in _ink_runs(words):
        pixels = transform.apply(run)
        if len(pixels) == 1:
            _stamp(image, pixels[:, 0], pixels[:, 1], radius)
            continue
        for start, end in zip(pixels[:-1], pixels[1:]):
            clipped = _clip_segment(start, end, transform.width, transform.height, radius)
            if clipped is None:
                continue
            clipped_start, clipped_end, _ = clipped
            samples = max(2, int(np.ceil(np.max(np.abs(clipped_end - clipped_start)) * 2.0)) + 1)
            alpha = np.linspace(0.0, 1.0, samples)
            points = clipped_start[None, :] + alpha[:, None] * (clipped_end - clipped_start)[None, :]
            _stamp(image, points[:, 0], points[:, 1], radius)
    return image


def _out_of_bounds_fraction(words: Sequence[np.ndarray], transform: CanvasTransform) -> float:
    total_length = 0.0
    inside_length = 0.0
    point_total = 0
    point_inside = 0
    for run in _ink_runs(words):
        pixels = transform.apply(run)
        inside = (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= transform.width - 1.0)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= transform.height - 1.0)
        )
        point_total += len(pixels)
        point_inside += int(inside.sum())
        for start, end in zip(pixels[:-1], pixels[1:]):
            length = float(np.linalg.norm(end - start))
            if length <= _EPS:
                continue
            total_length += length
            clipped = _clip_segment(start, end, transform.width, transform.height)
            if clipped is not None:
                inside_length += length * clipped[2]
    if total_length > _EPS:
        return float(np.clip(1.0 - inside_length / total_length, 0.0, 1.0))
    if point_total:
        return float(np.clip(1.0 - point_inside / point_total, 0.0, 1.0))
    return 0.0


def render_stroke_pair(
    prediction: Any,
    target: Any,
    *,
    config: RenderConfig | None = None,
    prediction_is_polar: bool = False,
    target_is_polar: bool = False,
) -> RenderedPair:
    """Render prediction and target on a target-controlled fixed canvas."""

    config = config or RenderConfig()
    prediction_polar_valid = True
    target_polar_valid = True
    if prediction_is_polar:
        try:
            prediction = polar_words_to_cartesian(prediction)
        except (TypeError, ValueError, IndexError):
            prediction = None
            prediction_polar_valid = False
    if target_is_polar:
        try:
            target = polar_words_to_cartesian(target)
        except (TypeError, ValueError, IndexError):
            target = None
            target_polar_valid = False

    pred_words, pred_valid, pred_reasons = _coerce_words(prediction, config.max_abs_coordinate)
    target_words, target_valid, target_reasons = _coerce_words(target, config.max_abs_coordinate)
    if not prediction_polar_valid:
        pred_valid = False
        pred_reasons = pred_reasons + ("invalid_polar_prediction",)
    if not target_polar_valid:
        target_valid = False
        target_reasons = target_reasons + ("invalid_polar_target",)
    pred_words = layout_words(pred_words, config.word_gap)
    target_words = layout_words(target_words, config.word_gap)
    transform = _fit_target_transform(target_words, config)
    target_image = rasterize_strokes(target_words, transform, config.line_width_px)
    pred_image = rasterize_strokes(pred_words, transform, config.line_width_px)
    return RenderedPair(
        prediction=pred_image,
        target=target_image,
        transform=transform,
        out_of_bounds_fraction=_out_of_bounds_fraction(pred_words, transform),
        prediction_valid=pred_valid,
        target_valid=target_valid,
        prediction_reasons=pred_reasons,
        target_reasons=target_reasons,
        prediction_words=pred_words,
        target_words=target_words,
    )


def _soft_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.maximum(first, second).sum(dtype=np.float64)
    if union <= _EPS:
        return 0.0
    intersection = np.minimum(first, second).sum(dtype=np.float64)
    return float(np.clip(intersection / union, 0.0, 1.0))


def _shape_scores(
    prediction: np.ndarray, target: np.ndarray, config: RenderConfig
) -> tuple[float, float, float, dict[str, float]]:
    pred_mask = prediction > 0.0
    target_mask = target > 0.0
    if not pred_mask.any() or not target_mask.any():
        scales = {f"ink_iou_sigma_{sigma:g}": 0.0 for sigma in config.blur_sigmas}
        return 0.0, 0.0, 0.0, scales

    overlaps: list[float] = []
    scale_scores: dict[str, float] = {}
    for sigma in config.blur_sigmas:
        if sigma > 0.0:
            pred_soft = gaussian_filter(prediction, sigma=sigma, mode="constant")
            target_soft = gaussian_filter(target, sigma=sigma, mode="constant")
        else:
            pred_soft, target_soft = prediction, target
        overlap = _soft_iou(pred_soft, target_soft)
        overlaps.append(overlap)
        scale_scores[f"ink_iou_sigma_{sigma:g}"] = overlap
    multiscale_overlap = float(np.mean(overlaps))

    distance_to_pred = distance_transform_edt(~pred_mask)
    distance_to_target = distance_transform_edt(~target_mask)
    target_distance = np.minimum(distance_to_pred[target_mask], config.distance_clip_px).mean()
    pred_distance = np.minimum(distance_to_target[pred_mask], config.distance_clip_px).mean()
    target_coverage = float(np.clip(1.0 - target_distance / config.distance_clip_px, 0.0, 1.0))
    prediction_precision = float(np.clip(1.0 - pred_distance / config.distance_clip_px, 0.0, 1.0))
    # A geometric mean prevents a tiny subset of the target from earning about
    # half credit merely because every predicted pixel happens to touch target ink.
    distance_similarity = float(np.sqrt(target_coverage * prediction_precision))
    scale_scores["distance_target_coverage"] = target_coverage
    scale_scores["distance_prediction_precision"] = prediction_precision
    shape_similarity = 0.55 * multiscale_overlap + 0.45 * distance_similarity
    return float(shape_similarity), multiscale_overlap, distance_similarity, scale_scores


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.average(np.asarray(values), weights=np.maximum(np.asarray(weights), _EPS)))


def stroke_style_statistics(strokes: Any, *, max_abs_coordinate: float = 1e5) -> dict[str, float]:
    """Compute interpretable, scale-aware handwriting style statistics."""

    words, valid, _ = _coerce_words(strokes, max_abs_coordinate)
    slants: list[float] = []
    slant_weights: list[float] = []
    curvatures: list[float] = []
    curvature_weights: list[float] = []
    aspect_ratios: list[float] = []
    densities: list[float] = []
    point_densities: list[float] = []
    lifts: list[float] = []
    stroke_counts: list[float] = []

    for word in words:
        runs = []
        start: int | None = None
        for index, point in enumerate(word):
            if point[2] > 0.5:
                if start is None:
                    start = index
            elif start is not None:
                runs.append(word[start:index, :2])
                start = None
        if start is not None:
            runs.append(word[start:, :2])
        runs = [run for run in runs if len(run)]
        if not runs:
            continue

        points = np.concatenate(runs, axis=0)
        spans = np.ptp(points, axis=0)
        width, height = max(float(spans[0]), _EPS), max(float(spans[1]), _EPS)
        aspect_ratios.append(width / height)
        stroke_counts.append(float(len(runs)))
        lifts.append(float(max(len(runs) - 1, 0)))

        path_length = 0.0
        point_count = 0
        for run in runs:
            point_count += len(run)
            if len(run) < 2:
                continue
            delta = np.diff(run, axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            useful = lengths > _EPS
            delta, lengths = delta[useful], lengths[useful]
            if not len(lengths):
                continue
            path_length += float(lengths.sum())

            vertical = np.abs(delta[:, 1]) >= 0.35 * lengths
            if vertical.any():
                direction = np.where(delta[vertical, 1] >= 0.0, 1.0, -1.0)
                angles = np.arctan2(delta[vertical, 0] * direction, np.abs(delta[vertical, 1]))
                slants.extend(angles.tolist())
                slant_weights.extend(lengths[vertical].tolist())

            if len(delta) >= 2:
                headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
                turns = np.minimum(np.abs(np.diff(headings)), np.pi)
                turn_weights = 0.5 * (lengths[:-1] + lengths[1:])
                curvatures.extend(turns.tolist())
                curvature_weights.extend(turn_weights.tolist())

        densities.append(path_length / max(width * height, _EPS))
        point_densities.append(point_count / max(path_length, _EPS))

    return {
        "valid": float(valid),
        "word_count": float(len(aspect_ratios)),
        "slant_rad": _weighted_mean(slants, slant_weights),
        "aspect_ratio": float(np.median(aspect_ratios)) if aspect_ratios else 0.0,
        "curvature_rad": _weighted_mean(curvatures, curvature_weights),
        "stroke_density": float(np.median(densities)) if densities else 0.0,
        "point_density": float(np.median(point_densities)) if point_densities else 0.0,
        "pen_lifts_per_word": float(np.mean(lifts)) if lifts else 0.0,
        "strokes_per_word": float(np.mean(stroke_counts)) if stroke_counts else 0.0,
    }


def _ratio_similarity(first: float, second: float, scale: float) -> float:
    if first <= _EPS or second <= _EPS:
        return float(first <= _EPS and second <= _EPS)
    return float(np.exp(-abs(np.log(first / second)) / scale))


def _style_scores(
    prediction: Sequence[np.ndarray], reference: Sequence[np.ndarray]
) -> tuple[float, dict[str, float]]:
    pred = stroke_style_statistics(prediction)
    ref = stroke_style_statistics(reference)
    if pred["word_count"] == 0.0 or ref["word_count"] == 0.0:
        similarities = {
            "style_slant_similarity": 0.0,
            "style_aspect_similarity": 0.0,
            "style_curvature_similarity": 0.0,
            "style_density_similarity": 0.0,
            "style_pen_lift_similarity": 0.0,
        }
    else:
        similarities = {
            "style_slant_similarity": float(
                np.exp(-abs(pred["slant_rad"] - ref["slant_rad"]) / 0.35)
            ),
            "style_aspect_similarity": _ratio_similarity(
                pred["aspect_ratio"], ref["aspect_ratio"], 0.60
            ),
            "style_curvature_similarity": float(
                np.exp(-abs(pred["curvature_rad"] - ref["curvature_rad"]) / 0.75)
            ),
            "style_density_similarity": _ratio_similarity(
                pred["stroke_density"], ref["stroke_density"], 0.80
            ),
            "style_pen_lift_similarity": float(
                np.exp(
                    -abs(pred["pen_lifts_per_word"] - ref["pen_lifts_per_word"])
                    / (1.0 + ref["pen_lifts_per_word"])
                )
            ),
        }
    metrics: dict[str, float] = dict(similarities)
    for name in (
        "slant_rad",
        "aspect_ratio",
        "curvature_rad",
        "stroke_density",
        "point_density",
        "pen_lifts_per_word",
        "strokes_per_word",
    ):
        metrics[f"pred_{name}"] = float(pred[name])
        metrics[f"reference_{name}"] = float(ref[name])
    return float(np.mean(list(similarities.values()))), metrics


def _word_count_score(predicted: int, expected: int) -> float:
    if expected <= 0:
        return 1.0 if predicted == 0 else -1.0
    error = min(abs(predicted - expected) / expected, 1.0)
    return float(1.0 - 2.0 * error)


def _length_score(ratio: float, config: RenderConfig) -> float:
    band_low, band_high = config.length_band
    limit_low, limit_high = config.length_limits
    if band_low <= ratio <= band_high:
        return 1.0
    if ratio < band_low:
        return float(np.clip(-1.0 + 2.0 * (ratio - limit_low) / (band_low - limit_low), -1.0, 1.0))
    return float(np.clip(1.0 - 2.0 * (ratio - band_high) / (limit_high - band_high), -1.0, 1.0))


def compute_rendering_reward(
    prediction: Any,
    target: Any,
    *,
    style_reference: Any | None = None,
    sequence: SequenceDiagnostics | None = None,
    predicted_tokens: Any | None = None,
    token_spec: TokenSpec | None = None,
    expected_word_count: int | None = None,
    target_token_length: int | None = None,
    config: RenderConfig | None = None,
    weights: RewardWeights | None = None,
    prediction_is_polar: bool = False,
    target_is_polar: bool = False,
    style_reference_is_polar: bool = False,
) -> RewardResult:
    """Compute a composite rendering-aware handwriting reward.

    ``sequence`` may be supplied directly, or inferred from ``predicted_tokens``
    plus ``token_spec``.  A style reference defaults to the target when omitted.
    Blank, truncated, malformed, extremely short, and almost-entirely-OOB
    generations are capped at ``config.hard_failure_ceiling`` so syntax bonuses
    cannot outweigh a broken rendering.
    """

    config = config or RenderConfig()
    weights = weights or RewardWeights()
    if sequence is None and predicted_tokens is not None:
        if token_spec is None:
            raise ValueError("token_spec is required with predicted_tokens")
        sequence = analyze_token_sequence(predicted_tokens, token_spec)

    rendered = render_stroke_pair(
        prediction,
        target,
        config=config,
        prediction_is_polar=prediction_is_polar,
        target_is_polar=target_is_polar,
    )
    shape, overlap, distance_score, scale_scores = _shape_scores(
        rendered.prediction, rendered.target, config
    )

    if style_reference is None:
        style_words = rendered.target_words
    else:
        if style_reference_is_polar:
            try:
                style_reference = polar_words_to_cartesian(style_reference)
            except (TypeError, ValueError, IndexError):
                style_reference = None
        style_words, _, _ = _coerce_words(style_reference, config.max_abs_coordinate)
    style, style_metrics = _style_scores(rendered.prediction_words, style_words)

    pred_ink_words = _ink_word_count(rendered.prediction_words)
    target_ink_words = _ink_word_count(rendered.target_words)
    predicted_words = (
        sequence.word_count
        if sequence is not None and sequence.word_count is not None
        else pred_ink_words
    )
    expected_words = target_ink_words if expected_word_count is None else expected_word_count
    word_score = _word_count_score(int(predicted_words), int(expected_words))

    pred_pixels = int(np.count_nonzero(rendered.prediction))
    target_pixels = int(np.count_nonzero(rendered.target))
    blank = pred_pixels < config.min_ink_pixels
    target_blank = target_pixels < config.min_ink_pixels
    ink_mass_ratio = pred_pixels / max(target_pixels, 1)
    geometric_collapse = not blank and ink_mass_ratio < config.minimum_ink_mass_ratio
    sequence_valid = sequence.valid if sequence is not None else True
    malformed = not rendered.prediction_valid or not sequence_valid
    ended = sequence.ended if sequence is not None else None
    truncated = ended is False

    length_ratio: float | None = None
    length = 0.0
    extremely_short = False
    if (
        target_token_length is not None
        and target_token_length > 0
        and sequence is not None
        and sequence.token_length is not None
    ):
        length_ratio = sequence.token_length / target_token_length
        length = _length_score(length_ratio, config)
        extremely_short = length_ratio < config.length_limits[0]

    oob = rendered.out_of_bounds_fraction
    hard_oob = oob >= config.hard_oob_fraction
    target_malformed = not rendered.target_valid
    hard_failure = (
        blank
        or geometric_collapse
        or target_blank
        or target_malformed
        or truncated
        or malformed
        or extremely_short
        or hard_oob
    )
    validity_score = 1.0 if (not malformed and rendered.target_valid and not target_blank) else -1.0

    components: dict[str, float] = {
        "shape_similarity": shape,
        "multiscale_ink_overlap": overlap,
        "distance_similarity": distance_score,
        **scale_scores,
        "style_similarity": style,
        **style_metrics,
        "word_count_score": word_score,
        "end_score": 0.0 if ended is None else (1.0 if ended else -1.0),
        "validity_score": validity_score,
        "out_of_bounds_penalty": -oob,
        "length_score": length,
        "blank_penalty": -1.0 if blank else 0.0,
        "hard_failure_penalty": -1.0 if hard_failure else 0.0,
        "ink_mass_ratio": float(ink_mass_ratio),
    }

    weighted: Mapping[str, tuple[float, float, bool]] = {
        "shape": (weights.shape, shape, True),
        "style": (weights.style, style, True),
        "word_count": (weights.word_count, word_score, True),
        "ended": (weights.ended, components["end_score"], ended is not None),
        "validity": (weights.validity, validity_score, True),
        "out_of_bounds": (weights.out_of_bounds, -oob, True),
        "length": (weights.length, length, length_ratio is not None),
        "blank": (weights.blank, components["blank_penalty"], True),
        "hard_failure": (weights.hard_failure, components["hard_failure_penalty"], True),
    }
    total = float(sum(weight * value for weight, value, active in weighted.values() if active))
    if target_blank or target_malformed:
        total = -1.0
    elif hard_failure:
        total = min(total, config.hard_failure_ceiling)

    diagnostics: dict[str, Any] = {
        "prediction_blank": blank,
        "target_blank": target_blank,
        "geometric_collapse": geometric_collapse,
        "target_malformed": target_malformed,
        "malformed": malformed,
        "truncated": truncated,
        "extremely_short": extremely_short,
        "hard_oob": hard_oob,
        "hard_failure": hard_failure,
        "prediction_word_count": int(predicted_words),
        "expected_word_count": int(expected_words),
        "prediction_ink_word_count": pred_ink_words,
        "target_ink_word_count": target_ink_words,
        "prediction_ink_pixels": pred_pixels,
        "target_ink_pixels": target_pixels,
        "out_of_bounds_fraction": oob,
        "length_ratio": length_ratio if length_ratio is not None else 0.0,
        "sequence_reasons": sequence.reasons if sequence is not None else (),
        "prediction_reasons": rendered.prediction_reasons,
        "target_reasons": rendered.target_reasons,
        "active_reward_terms": tuple(name for name, (_, _, active) in weighted.items() if active),
    }
    return RewardResult(total=total, components=components, diagnostics=diagnostics)


__all__ = [
    "CanvasTransform",
    "RenderConfig",
    "RenderedPair",
    "RewardResult",
    "RewardWeights",
    "SequenceDiagnostics",
    "TokenSpec",
    "analyze_token_sequence",
    "compute_rendering_reward",
    "layout_words",
    "polar_offsets_to_cartesian",
    "polar_words_to_cartesian",
    "rasterize_strokes",
    "render_stroke_pair",
    "stroke_style_statistics",
]
