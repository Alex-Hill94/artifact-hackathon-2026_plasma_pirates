"""Request and response shapes for the search API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

CompareOp = Literal[
    "eq", "ne", "lt", "lte", "gt", "gte", "between", "outside", "in", "isnan", "notnan"
]
BoolOp = Literal["and", "or"]

#: Operators and how many values they consume.
OP_ARITY: dict[str, int] = {
    "eq": 1,
    "ne": 1,
    "lt": 1,
    "lte": 1,
    "gt": 1,
    "gte": 1,
    "between": 2,
    "outside": 2,
    "in": -1,  # any number
    "isnan": 0,
    "notnan": 0,
}


class TimeRange(BaseModel):
    """A half-open window ``[from, to)``.

    ``days`` is an alternative to ``to`` so that "the 5 days following
    February 2nd" is expressible without the caller doing date arithmetic.
    Bounds given the wrong way round are swapped rather than rejected.
    """

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    days: float | None = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _resolve(self) -> "TimeRange":
        if self.days is not None:
            if self.from_ is None:
                raise ValueError("'days' requires 'from'")
            if self.to is not None:
                raise ValueError("give either 'to' or 'days', not both")
            # Collapse to an absolute bound and drop `days`, so re-validating an
            # already-resolved TimeRange is a no-op rather than an error.
            self.__dict__["to"] = self.from_ + timedelta(days=self.days)
            self.__dict__["days"] = None
        if self.from_ and self.to and self.from_ > self.to:
            self.__dict__["from_"], self.__dict__["to"] = self.to, self.from_
        return self


class CameraFilter(BaseModel):
    """Cameras by exact name and/or by the area segment of their name."""

    names: list[str] = Field(default_factory=list)
    areas: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.names and not self.areas


class Clause(BaseModel):
    """A predicate on one PV, or on a glob-matched group of PVs."""

    pv: str | None = None
    pv_pattern: str | None = None
    match: Literal["any", "all"] = "any"
    op: CompareOp
    value: Any = None
    #: Absolute tolerance for ``eq``/``ne``. These are floats read back from a
    #: control system; exact equality is rarely what the user means.
    tol: float = 1e-9

    @model_validator(mode="after")
    def _check(self) -> "Clause":
        if (self.pv is None) == (self.pv_pattern is None):
            raise ValueError("give exactly one of 'pv' or 'pv_pattern'")
        arity = OP_ARITY[self.op]
        values = self.value
        if arity == 0:
            return self
        if arity == 1:
            if isinstance(values, (list, tuple)):
                raise ValueError(f"operator '{self.op}' takes a single value")
            if values is None:
                raise ValueError(f"operator '{self.op}' requires a value")
        elif arity == 2:
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"operator '{self.op}' requires exactly two values")
        elif arity == -1:
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError("operator 'in' requires a non-empty list")
        return self


class Group(BaseModel):
    """A nestable boolean combination of clauses."""

    op: BoolOp = "and"
    clauses: list["Node"] = Field(default_factory=list)


Node = Annotated[Union[Clause, Group], Field(union_mode="left_to_right")]
Group.model_rebuild()


class Sort(BaseModel):
    by: Literal["timestamp", "camera"] = "timestamp"
    order: Literal["asc", "desc"] = "asc"


class SearchRequest(BaseModel):
    time: TimeRange = Field(default_factory=TimeRange)
    cameras: CameraFilter = Field(default_factory=CameraFilter)
    filters: Group | None = None
    #: PVs whose values to include with each returned row.
    select: list[str] = Field(default_factory=list)
    sort: Sort = Field(default_factory=Sort)
    limit: int = Field(default=100, ge=0)
    offset: int = Field(default=0, ge=0)


class SeriesRequest(BaseModel):
    """A time series of one or more PVs, for plotting."""

    pvs: list[str] = Field(default_factory=list)
    pv_pattern: str | None = None
    time: TimeRange = Field(default_factory=TimeRange)
    cameras: CameraFilter = Field(default_factory=CameraFilter)
    filters: Group | None = None
    #: Series longer than this are min/max decimated server-side.
    max_points: int = Field(default=4000, ge=2, le=200_000)

    @model_validator(mode="after")
    def _check(self) -> "SeriesRequest":
        if not self.pvs and not self.pv_pattern:
            raise ValueError("give 'pvs' and/or 'pv_pattern'")
        return self


# --- responses ---------------------------------------------------------------


class RowOut(BaseModel):
    id: int
    timestamp: str
    camera: str
    area: str | None = None
    capture_path: str
    n_images: int
    source_file: str
    #: Only ever true on a single-row fetch; listings do not touch the disk.
    has_image: bool = False
    values: dict[str, float | None] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    total: int
    limit: int
    offset: int
    elapsed_ms: float
    #: Rows the time window narrowed to, i.e. how much the query actually touched.
    scanned_rows: int
    #: Hyperslab reads issued against the backend.
    pv_reads: int
    rows: list[RowOut]


class PVOut(BaseModel):
    name: str
    key: str
    machine: str | None = None
    area: str | None = None
    family: str | None = None
    device: str | None = None
    number: str | None = None
    field: str | None = None
    min: float | None = None
    max: float | None = None


class CameraOut(BaseModel):
    name: str
    area: str | None = None
    n_rows: int
    first: str | None = None
    last: str | None = None


class MetaOut(BaseModel):
    domain: str
    schema_version: int
    created_utc: str | None
    n_rows: int
    pv_count: int
    time_from: str | None
    time_to: str | None
    cameras: list[CameraOut]


class SeriesOut(BaseModel):
    pv: str
    #: ISO-8601 UTC timestamps, aligned with ``v``.
    t: list[str]
    v: list[float | None]
    decimated: bool


class SeriesResponse(BaseModel):
    total: int
    elapsed_ms: float
    series: list[SeriesOut]
