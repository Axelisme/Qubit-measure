"""Formula recipes, restricted expressions and waveform rendering."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from ._waveform_data import (
    MAX_ARB_WAVEFORM_SAMPLES,
    ArbWaveformData,
    ArbWaveformError,
    coerce_positive_float,
    peak_abs,
    validate_payload,
)

ARB_WAVEFORM_RENDER_SAMPLES_PER_US = 1000.0


_ALLOWED_FUNCTIONS = frozenset(
    {"sin", "cos", "tan", "exp", "sqrt", "Abs", "abs", "erf"}
)


_ALLOWED_NAMES = frozenset({"t", "T", "pi", "e", "I", *_ALLOWED_FUNCTIONS})


NormalizeMode = Literal["none", "peak"]


@dataclass(frozen=True)
class FormulaSegment:
    duration: float
    formula: str

    @classmethod
    def from_raw(cls, raw: object, *, index: int) -> FormulaSegment:
        if not isinstance(raw, dict):
            raise ArbWaveformError(
                f"segments[{index}] must be an object",
                reason="invalid_recipe",
                data={"path": f"segments[{index}]"},
            )
        extra = set(raw) - {"duration", "formula"}
        if extra:
            raise ArbWaveformError(
                f"segments[{index}] has unsupported key(s): {sorted(extra)}",
                reason="invalid_recipe",
                data={"path": f"segments[{index}]", "unknown_keys": sorted(extra)},
            )
        if "duration" not in raw:
            raise ArbWaveformError(
                f"segments[{index}].duration is required",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].duration"},
            )
        if "formula" not in raw:
            raise ArbWaveformError(
                f"segments[{index}].formula is required",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].formula"},
            )
        duration = coerce_positive_float(
            raw["duration"], path=f"segments[{index}].duration"
        )
        formula = raw["formula"]
        if not isinstance(formula, str):
            raise ArbWaveformError(
                f"segments[{index}].formula must be a string",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].formula"},
            )
        if not formula.strip():
            raise ArbWaveformError(
                f"segments[{index}].formula must not be empty",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].formula"},
            )
        return cls(duration=duration, formula=formula)

    def to_dict(self) -> dict[str, object]:
        return {"duration": self.duration, "formula": self.formula}


@dataclass(frozen=True)
class FormulaRecipe:
    segments: tuple[FormulaSegment, ...]
    normalize: NormalizeMode

    @classmethod
    def from_raw(cls, raw: object) -> FormulaRecipe:
        if isinstance(raw, FormulaRecipe):
            return raw
        if not isinstance(raw, dict):
            raise ArbWaveformError(
                "recipe must be an object",
                reason="invalid_recipe",
                data={"path": ""},
            )
        extra = set(raw) - {"segments", "normalize"}
        if extra:
            raise ArbWaveformError(
                f"recipe has unsupported key(s): {sorted(extra)}",
                reason="invalid_recipe",
                data={"path": "", "unknown_keys": sorted(extra)},
            )
        if "segments" not in raw:
            raise ArbWaveformError(
                "recipe.segments is required",
                reason="invalid_recipe",
                data={"path": "segments"},
            )
        if "normalize" not in raw:
            raise ArbWaveformError(
                "recipe.normalize is required",
                reason="invalid_recipe",
                data={"path": "normalize"},
            )
        segments_raw = raw["segments"]
        if not isinstance(segments_raw, list):
            raise ArbWaveformError(
                "recipe.segments must be a list",
                reason="invalid_recipe",
                data={"path": "segments"},
            )
        if not segments_raw:
            raise ArbWaveformError(
                "recipe.segments must contain at least one segment",
                reason="invalid_recipe",
                data={"path": "segments"},
            )
        normalize = raw["normalize"]
        if normalize not in ("none", "peak"):
            raise ArbWaveformError(
                "recipe.normalize must be 'none' or 'peak'",
                reason="invalid_recipe",
                data={"path": "normalize"},
            )
        return cls(
            segments=tuple(
                FormulaSegment.from_raw(item, index=index)
                for index, item in enumerate(segments_raw)
            ),
            normalize=normalize,
        )

    @classmethod
    def from_json(cls, text: str) -> FormulaRecipe:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArbWaveformError(
                f"recipe_json is not valid JSON: {exc.msg}",
                reason="invalid_recipe_json",
                data={"pos": exc.pos},
            ) from exc
        try:
            return cls.from_raw(raw)
        except ArbWaveformError as exc:
            if exc.reason == "invalid_recipe":
                raise ArbWaveformError(
                    str(exc), reason="invalid_recipe_json", data=exc.data
                ) from exc
            raise

    def to_dict(self) -> dict[str, object]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "normalize": self.normalize,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def render_formula_recipe(recipe: FormulaRecipe | dict[str, object]) -> ArbWaveformData:
    parsed = FormulaRecipe.from_raw(recipe)
    total_duration = math.fsum(segment.duration for segment in parsed.segments)
    sample_count = round(total_duration * ARB_WAVEFORM_RENDER_SAMPLES_PER_US) + 1
    if sample_count < 2:
        raise ArbWaveformError(
            "formula duration is too short for the internal render resolution",
            reason="sample_count_too_small",
            data={"sample_count": sample_count, "total_duration": total_duration},
        )
    if sample_count > MAX_ARB_WAVEFORM_SAMPLES:
        raise ArbWaveformError(
            f"formula render would produce {sample_count} samples, exceeding "
            f"{MAX_ARB_WAVEFORM_SAMPLES}",
            reason="sample_count_too_large",
            data={
                "sample_count": sample_count,
                "max_sample_count": MAX_ARB_WAVEFORM_SAMPLES,
                "total_duration": total_duration,
            },
        )

    time = np.linspace(0.0, total_duration, sample_count, dtype=np.float64)
    boundaries = np.cumsum(
        [segment.duration for segment in parsed.segments], dtype=np.float64
    )
    segment_index = np.searchsorted(boundaries, time, side="right")
    segment_index = np.minimum(segment_index, len(parsed.segments) - 1)
    starts = np.concatenate(([0.0], boundaries[:-1]))

    idata = np.zeros(sample_count, dtype=np.float64)
    qdata = np.zeros(sample_count, dtype=np.float64)
    for index, segment in enumerate(parsed.segments):
        mask = segment_index == index
        if not np.any(mask):
            continue
        local_t = time[mask] - starts[index]
        values = _evaluate_segment(
            segment.formula, local_t=local_t, global_t=time[mask]
        )
        try:
            values = np.broadcast_to(values, local_t.shape)
        except ValueError as exc:
            raise ArbWaveformError(
                f"segments[{index}].formula output cannot broadcast to segment samples",
                reason="formula_shape_mismatch",
                data={
                    "path": f"segments[{index}].formula",
                    "segment_index": index,
                    "output_shape": values.shape,
                    "sample_shape": local_t.shape,
                },
            ) from exc
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ArbWaveformError(
                f"segments[{index}].formula produced non-finite values",
                reason="formula_non_finite",
                data={"path": f"segments[{index}].formula", "segment_index": index},
            )
        idata[mask] = values.real.astype(np.float64, copy=False)
        qdata[mask] = values.imag.astype(np.float64, copy=False)

    peak = peak_abs(idata, qdata)
    if parsed.normalize == "peak" and peak > 0.0:
        idata = idata / peak
        qdata = qdata / peak

    return validate_payload(idata=idata, qdata=qdata, time=time, recipe=parsed)


def _evaluate_segment(
    formula: str, *, local_t: NDArray[np.float64], global_t: NDArray[np.float64]
) -> NDArray[np.complex128]:
    expr = _parse_formula(formula)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    func = sp.lambdify(
        (_SYM_T, _SYM_GLOBAL_T),
        expr,
        modules=[{"Abs": np.abs, "erf": erf_vec}, "numpy"],
    )
    try:
        raw = func(local_t, global_t)
    except Exception as exc:
        raise ArbWaveformError(
            f"formula evaluation failed: {exc}",
            reason="formula_evaluation_failed",
            data={"formula": formula},
        ) from exc
    try:
        arr = np.asarray(raw, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ArbWaveformError(
            "formula output must be numeric",
            reason="formula_not_numeric",
            data={"formula": formula},
        ) from exc
    return arr


_SYM_T = sp.Symbol("t")


_SYM_GLOBAL_T = sp.Symbol("T")


_LOCAL_DICT: dict[str, object] = {
    "t": _SYM_T,
    "T": _SYM_GLOBAL_T,
    "pi": sp.pi,
    "e": sp.E,
    "I": sp.I,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "erf": sp.erf,
}


_GLOBAL_DICT: dict[str, object] = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
}


def _parse_formula(formula: str) -> sp.Expr:
    stripped = formula.strip()
    _validate_formula_ast(stripped)
    try:
        expr = parse_expr(
            stripped,
            local_dict=_LOCAL_DICT,
            global_dict=_GLOBAL_DICT,
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise ArbWaveformError(
            f"formula parse failed: {exc}",
            reason="formula_parse_failed",
            data={"formula": formula},
        ) from exc
    if not isinstance(expr, sp.Expr):
        raise ArbWaveformError(
            "formula must parse to a SymPy expression",
            reason="formula_parse_failed",
            data={"formula": formula},
        )
    symbols = {str(symbol) for symbol in expr.free_symbols}
    unsupported = symbols - {"t", "T"}
    if unsupported:
        raise ArbWaveformError(
            f"formula uses unsupported symbol(s): {sorted(unsupported)}",
            reason="formula_unknown_symbol",
            data={"symbols": sorted(unsupported)},
        )
    if expr.has(sp.Piecewise) or expr.atoms(sp.core.relational.Relational):
        raise ArbWaveformError(
            "formula conditionals and comparisons are not supported",
            reason="formula_conditional_not_supported",
            data={"formula": formula},
        )
    return expr


def _validate_formula_ast(formula: str) -> None:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ArbWaveformError(
            f"formula syntax error: {exc.msg}",
            reason="formula_parse_failed",
            data={"formula": formula},
        ) from exc
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.UAdd,
                ast.USub,
            ),
        ):
            continue
        raise ArbWaveformError(
            f"formula syntax is not supported: {type(node).__name__}",
            reason="formula_syntax_not_supported",
            data={"node": type(node).__name__},
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ArbWaveformError(
                f"formula uses unsupported name {node.id!r}",
                reason="formula_unknown_symbol",
                data={"symbol": node.id},
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCTIONS
            ):
                name = getattr(node.func, "id", type(node.func).__name__)
                raise ArbWaveformError(
                    f"formula uses unsupported function {name!r}",
                    reason="formula_unknown_function",
                    data={"function": name},
                )
            if node.keywords:
                raise ArbWaveformError(
                    "formula function calls do not support keyword arguments",
                    reason="formula_syntax_not_supported",
                    data={"node": "keyword"},
                )
