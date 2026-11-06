import json
import logging
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional, Union

import pandas as pd

from shared import usage
from shared.models.chart import COLLECTION as CHARTS, Chart
from shared.storage import build_chart_key, get_bucket_name, get_s3_client, new_file_id

logger = logging.getLogger("shared.services.visualization")

PALETTE = [
    "#CC785C",
    "#4A7C7C",
    "#D9A566",
    "#8B6F9E",
    "#B0562B",
    "#6B8F71",
    "#C4956C",
    "#5F7A99",
    "#A8754A",
    "#D97757",
]

VALID_CHART_TYPES = {
    "bar", "line", "timeline", "scatter3d", "surface", "pie", "histogram", "box", "heatmap",
}

_CHART_TYPE_ALIASES = {
    "donut": "pie",
    "doughnut": "pie",
    "scatter": "scatter3d",
    "bubble": "scatter3d",
}


@dataclass
class StagedChart:
    chart_id: str
    workspace_id: str
    title: str
    chart_type: str
    location: str  # local path to rendered HTML file
    artifact_file_id: Optional[str] = None
    spec: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VisualizationService:
    """
    Dedicated unified service to handle all chart operations across the platform:
    1. propose_visualization: validates and defines chart specifications.
    2. create_visualization: renders interactive HTML, writes local artifacts, stages by ID.
    3. stage_chart: records staged chart metadata and keeps full payloads out of LLM context.
    4. persist_chart / persist_all: uploads to S3, stores in MongoDB, updates user usage.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        self._staged_charts: dict[str, StagedChart] = {}
        os.makedirs(output_dir, exist_ok=True)

    def propose_visualization(
        self,
        chart_type: str,
        title: str,
        x_axis: Optional[str] = None,
        y_axis: Optional[str] = None,
        category: Optional[str] = None,
        label_column: Optional[str] = None,
        value_columns: Optional[list[str]] = None,
        time_column: Optional[str] = None,
        series_column: Optional[str] = None,
        value_column: Optional[str] = None,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        z_column: Optional[str] = None,
        bins: Optional[int] = None,
        file_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        available_columns: Optional[list[str]] = None,
        **extra_meta: Any,
    ) -> dict[str, Any]:
        """
        Validate and propose a visualization specification without rendering it.
        Returns a structured proposal with a unique chart_id and normalized parameters.
        """
        chart_type_norm = (chart_type or "bar").lower().strip()
        chart_type_norm = _CHART_TYPE_ALIASES.get(chart_type_norm, chart_type_norm)

        if chart_type_norm not in VALID_CHART_TYPES:
            return {
                "status": "error",
                "error": f"Unsupported chart_type '{chart_type}'. Choose from: {sorted(VALID_CHART_TYPES)}",
            }

        # Normalize x/y/category mappings into standard spec attributes
        final_label = label_column or x_axis
        final_x = x_column or x_axis
        final_time = time_column or (x_axis if chart_type_norm == "timeline" else None)

        final_val_col = value_column or y_axis
        final_val_cols = list(value_columns) if value_columns else ([y_axis] if y_axis else [])
        final_y = y_column or y_axis

        final_series = series_column or category
        final_z = z_column or (category if chart_type_norm in ("scatter3d", "surface", "heatmap") else None)

        # Validate column names against available_columns if provided
        if available_columns:
            needed_cols = [c for c in [final_label, final_x, final_time, final_val_col, final_series, final_y, final_z] if c]
            needed_cols.extend(final_val_cols)
            missing = [c for c in needed_cols if c not in available_columns]
            if missing:
                return {
                    "status": "error",
                    "error": f"Columns not found in dataset: {missing}. Available columns: {available_columns}",
                }

        cid = f"chart_{chart_type_norm}_{uuid.uuid4().hex[:10]}"
        spec = {
            "chart_id": cid,
            "chart_type": chart_type_norm,
            "title": title or "Untitled Chart",
            "label_column": final_label,
            "value_columns": final_val_cols,
            "time_column": final_time,
            "series_column": final_series,
            "value_column": final_val_col,
            "x_column": final_x,
            "y_column": final_y,
            "z_column": final_z,
            "bins": bins,
            "file_id": file_id or "data",
            "connection_id": connection_id,
            **extra_meta,
        }

        return {
            "status": "proposed",
            "chart_id": cid,
            "title": spec["title"],
            "chart_type": chart_type_norm,
            "specification": spec,
            "message": f"Proposed {chart_type_norm} visualization '{title}' with chart_id '{cid}'.",
        }

    def create_visualization(
        self,
        spec: Optional[dict] = None,
        data: Optional[Union[pd.DataFrame, list[dict]]] = None,
        workspace_id: str = "default",
        chart_id: Optional[str] = None,
        title: Optional[str] = None,
        chart_type: Optional[str] = None,
        file_id: Optional[str] = None,
        storage_root: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create, render, and stage a single chart as a local artifact.
        Returns ONLY compact metadata + chart_id to keep the LLM turn lean.
        """
        spec = dict(spec) if spec else {}
        spec.update({k: v for k, v in kwargs.items() if v is not None})
        if title:
            spec["title"] = title
        if chart_type:
            spec["chart_type"] = chart_type
        if file_id:
            spec["file_id"] = file_id
        if chart_id:
            spec["chart_id"] = chart_id

        # Normalize chart type
        raw_type = (spec.get("chart_type") or "bar").lower().strip()
        spec["chart_type"] = _CHART_TYPE_ALIASES.get(raw_type, raw_type)
        if spec["chart_type"] not in VALID_CHART_TYPES:
            raise ValueError(f"Unsupported chart_type '{raw_type}'. Must be one of {sorted(VALID_CHART_TYPES)}")

        # Reconcile x_axis/y_axis if present in kwargs
        if "x_axis" in kwargs and not spec.get("label_column"):
            spec["label_column"] = kwargs["x_axis"]
            spec["x_column"] = kwargs["x_axis"]
        if "y_axis" in kwargs and not spec.get("value_columns"):
            spec["value_columns"] = [kwargs["y_axis"]]
            spec["value_column"] = kwargs["y_axis"]
            spec["y_column"] = kwargs["y_axis"]
        if "category" in kwargs and not spec.get("series_column"):
            spec["series_column"] = kwargs["category"]
            spec["z_column"] = kwargs["category"]

        cid = spec.get("chart_id") or f"chart_{spec['chart_type']}_{uuid.uuid4().hex[:10]}"
        spec["chart_id"] = cid
        chart_title = spec.get("title") or "Untitled Chart"

        # Resolve DataFrame from input data
        df: Optional[pd.DataFrame] = None
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif spec.get("file_id") and storage_root:
            fid = spec["file_id"]
            parquet_path = os.path.join(storage_root, workspace_id, f"{fid}.parquet")
            if not os.path.isfile(parquet_path):
                parquet_path = os.path.join(storage_root, workspace_id, fid)
            if os.path.isfile(parquet_path):
                df = pd.read_parquet(parquet_path)

        if df is None:
            raise ValueError("No data provided to create_visualization (must provide DataFrame, list of row dicts, or valid file_id)")

        # Render section and HTML
        section = self._render_section(df, spec)
        html_content = self._render_html(chart_title, [section], source_count=1)

        safe_title = re.sub(r"[^0-9a-zA-Z_-]", "_", chart_title)[:40].strip("_") or "chart"
        chart_folder = os.path.join(self.output_dir, date.today().isoformat(), f"{cid}_{safe_title}")
        os.makedirs(chart_folder, exist_ok=True)
        chart_path = os.path.join(chart_folder, "chart.html")

        with open(chart_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        staged_meta = self.stage_chart(
            workspace_id=workspace_id,
            title=chart_title,
            chart_type=spec["chart_type"],
            location=chart_path,
            chart_id=cid,
            artifact_file_id=spec.get("file_id"),
            spec=spec,
        )

        return {
            "status": "created",
            "chart_id": cid,
            "title": staged_meta["title"],
            "chart_type": staged_meta["chart_type"],
            "artifact_file_id": staged_meta.get("artifact_file_id"),
            "location": chart_path,
            "specification": spec,
        }

    def create_visualizations(
        self,
        visualizations: list[dict],
        workspace_id: str = "default",
        data_map: Optional[dict[str, Any]] = None,
        storage_root: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Batch render multiple visualization specifications.
        """
        charts = []
        errors = []
        data_map = data_map or {}

        for index, raw in enumerate(visualizations or []):
            try:
                fid = raw.get("file_id")
                data = data_map.get(fid) if fid else None
                created = self.create_visualization(
                    spec=raw,
                    data=data,
                    workspace_id=workspace_id,
                    storage_root=storage_root,
                )
                charts.append(created)
            except Exception as exc:
                logger.exception("Failed to create visualization %d: %s", index, exc)
                errors.append({"index": index, "error": str(exc), "spec": raw})

        status = "success" if charts and not errors else ("partial_error" if charts and errors else "error")
        return {"status": status, "charts": charts, "errors": errors}

    def stage_chart(
        self,
        workspace_id: str,
        title: str,
        chart_type: str,
        location: str,
        chart_id: Optional[str] = None,
        artifact_file_id: Optional[str] = None,
        spec: Optional[dict] = None,
        **extra_meta: Any,
    ) -> dict:
        """
        Stage a locally generated chart by ID.
        Returns ONLY compact metadata to keep the LLM context window small.
        """
        cid = chart_id or f"chart_{chart_type}_{uuid.uuid4().hex[:10]}"
        staged = StagedChart(
            chart_id=cid,
            workspace_id=workspace_id,
            title=title or "Untitled chart",
            chart_type=chart_type,
            location=location,
            artifact_file_id=artifact_file_id,
            spec=spec,
            metadata=extra_meta,
        )
        self._staged_charts[cid] = staged
        logger.info("Staged chart %s (%s) at %s", cid, title, location)

        return {
            "chart_id": cid,
            "title": staged.title,
            "chart_type": staged.chart_type,
            "artifact_file_id": staged.artifact_file_id,
            "status": "staged",
        }

    def get_by_id(self, chart_id: str) -> Optional[StagedChart]:
        """Fetch full staged chart record by its unique ID."""
        return self._staged_charts.get(chart_id)

    def list_staged(self, workspace_id: Optional[str] = None) -> list[StagedChart]:
        """List all staged charts, optionally filtered by workspace."""
        if workspace_id:
            return [c for c in self._staged_charts.values() if c.workspace_id == workspace_id]
        return list(self._staged_charts.values())

    async def persist_chart(
        self,
        chart_id: str,
        db,
        workspace_id: str,
        message_id: str,
        user_id: str,
        email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist a single staged chart to S3 and MongoDB.
        Enforces user usage quotas and cleans up local temporary files.
        """
        staged = self.get_by_id(chart_id)
        if staged is None:
            logger.warning("persist_chart: chart_id %s not found in staged charts", chart_id)
            return None

        ref = staged.location
        if not isinstance(ref, str) or not os.path.isfile(ref):
            logger.warning("persist_chart: local file %s missing for chart %s", ref, chart_id)
            return None

        if not await usage.has_chart_capacity(user_id, email):
            logger.info("persist_chart: user %s reached chart capacity", user_id)
            return None

        s3 = get_s3_client()
        bucket = get_bucket_name()
        persisted_id = new_file_id()
        key = build_chart_key(workspace_id, persisted_id)

        try:
            s3.upload_file(ref, bucket, key, ExtraArgs={"ContentType": "text/html"})
            chart_doc = Chart(
                id=persisted_id,
                workspace_id=workspace_id,
                message_id=message_id,
                title=staged.title,
                storage_key=key,
            )
            await db[CHARTS].insert_one(chart_doc.to_mongo())
            await usage.increment_charts(user_id)
            logger.info("Successfully persisted chart %s -> %s (key=%s)", chart_id, persisted_id, key)
            return persisted_id
        except Exception:
            logger.exception("Failed to upload/persist chart %s (%s)", chart_id, ref)
            return None
        finally:
            shutil.rmtree(os.path.dirname(ref), ignore_errors=True)
            self._staged_charts.pop(chart_id, None)

    async def persist_all(
        self,
        db,
        workspace_id: str,
        message_id: str,
        user_id: str,
        email: Optional[str] = None,
    ) -> list[str]:
        """Persist all currently staged charts to S3/MongoDB."""
        saved_ids: list[str] = []
        for cid in list(self._staged_charts.keys()):
            saved_id = await self.persist_chart(cid, db, workspace_id, message_id, user_id, email)
            if saved_id:
                saved_ids.append(saved_id)
        return saved_ids

    def clear(self) -> None:
        """Clear all staged charts from memory."""
        self._staged_charts.clear()

    # -------------------------------------------------------------------------
    # Chart Rendering Engine (Internal HTML & Plot generation)
    # -------------------------------------------------------------------------

    @staticmethod
    def _safe(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @classmethod
    def _pivot_datasets(cls, df: pd.DataFrame, index_col: str, series_col: str, value_col: str) -> tuple:
        pivot = df.pivot_table(index=index_col, columns=series_col, values=value_col, aggfunc="sum")
        labels = [str(v) for v in pivot.index.tolist()]
        datasets = [
            {"label": str(col), "data": [cls._safe(v) for v in pivot[col].tolist()]}
            for col in pivot.columns
        ]
        return labels, datasets

    @classmethod
    def _render_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        chart_type = spec.get("chart_type", "bar")
        if chart_type in ("bar", "line"):
            return cls._categorical_section(dataframe, spec)
        if chart_type == "timeline":
            return cls._timeline_section(dataframe, spec)
        if chart_type in ("scatter3d", "surface"):
            return cls._chart3d_section(dataframe, spec)
        if chart_type == "pie":
            return cls._pie_section(dataframe, spec)
        if chart_type == "histogram":
            return cls._histogram_section(dataframe, spec)
        if chart_type == "box":
            return cls._box_section(dataframe, spec)
        if chart_type == "heatmap":
            return cls._heatmap_section(dataframe, spec)
        raise ValueError(f"Unknown chart_type '{chart_type}'")

    @classmethod
    def _categorical_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        series_column = spec.get("series_column")
        value_column = spec.get("value_column")
        label_column = spec.get("label_column")
        value_columns = spec.get("value_columns") or []

        if series_column and value_column:
            if not label_column:
                label_column = dataframe.columns[0]
            labels, datasets = cls._pivot_datasets(dataframe, label_column, series_column, value_column)
            x_label = label_column
            y_label = value_column
        else:
            numeric_cols = [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
            label_col = label_column or next((c for c in dataframe.columns if c not in numeric_cols), dataframe.columns[0])
            val_cols = value_columns or numeric_cols[:5]
            if not val_cols and numeric_cols:
                val_cols = [numeric_cols[0]]
            rows = dataframe.head(50)
            labels = rows[label_col].astype(str).tolist()
            datasets = [
                {"label": col, "data": [cls._safe(v) for v in rows[col].tolist()]}
                for col in val_cols
            ]
            x_label = label_col
            y_label = val_cols[0] if len(val_cols) == 1 else None

        return {
            "kind": "chartjs",
            "chart_type": spec.get("chart_type", "bar"),
            "title": spec.get("title", "Chart"),
            "source": str(spec.get("file_id", "data")),
            "row_count": len(dataframe),
            "x_label": x_label,
            "y_label": y_label,
            "labels": labels,
            "datasets": datasets,
        }

    @classmethod
    def _timeline_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        time_column = spec.get("time_column") or spec.get("label_column") or dataframe.columns[0]
        df = dataframe.sort_values(time_column)
        series_column = spec.get("series_column")
        value_column = spec.get("value_column")

        if series_column and value_column:
            labels, datasets = cls._pivot_datasets(df, time_column, series_column, value_column)
            y_label = value_column
        else:
            value_cols = spec.get("value_columns") or [
                c for c in df.columns if c != time_column and pd.api.types.is_numeric_dtype(df[c])
            ][:5]
            labels = df[time_column].astype(str).tolist()
            datasets = [
                {"label": col, "data": [cls._safe(v) for v in df[col].tolist()]}
                for col in value_cols
            ]
            y_label = value_cols[0] if len(value_cols) == 1 else None

        return {
            "kind": "chartjs",
            "chart_type": "line",
            "title": spec.get("title", "Timeline"),
            "source": str(spec.get("file_id", "data")),
            "row_count": len(df),
            "x_label": time_column,
            "y_label": y_label,
            "labels": labels,
            "datasets": datasets,
        }

    @classmethod
    def _chart3d_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        x_col = spec.get("x_column") or dataframe.columns[0]
        y_col = spec.get("y_column") or (dataframe.columns[1] if len(dataframe.columns) > 1 else dataframe.columns[0])
        z_col = spec.get("z_column") or (dataframe.columns[2] if len(dataframe.columns) > 2 else y_col)

        df = dataframe.head(2000)
        title = spec.get("title", "3D Chart")
        chart_type = spec.get("chart_type", "scatter3d")

        if chart_type == "surface":
            pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
            z_matrix = [[cls._safe(v) for v in row] for row in pivot.values.tolist()]
            return {
                "kind": "plotly",
                "plot_type": "surface",
                "title": title,
                "source": str(spec.get("file_id", "data")),
                "row_count": len(df),
                "x_label": x_col,
                "y_label": y_col,
                "z_label": z_col,
                "x": [str(c) for c in pivot.columns.tolist()],
                "y": [str(i) for i in pivot.index.tolist()],
                "z": z_matrix,
            }

        return {
            "kind": "plotly",
            "plot_type": "scatter3d",
            "title": title,
            "source": str(spec.get("file_id", "data")),
            "row_count": len(df),
            "x_label": x_col,
            "y_label": y_col,
            "z_label": z_col,
            "x": [cls._safe(v) for v in df[x_col].tolist()],
            "y": [cls._safe(v) for v in df[y_col].tolist()],
            "z": [cls._safe(v) for v in df[z_col].tolist()],
        }

    @classmethod
    def _pie_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        label_col = spec.get("label_column") or dataframe.columns[0]
        val_col = spec.get("value_column") or (spec.get("value_columns") or [None])[0] or dataframe.columns[-1]
        rows = dataframe.head(20)
        return {
            "kind": "chartjs",
            "chart_type": "pie",
            "title": spec.get("title", "Pie Chart"),
            "source": str(spec.get("file_id", "data")),
            "row_count": len(dataframe),
            "x_label": label_col,
            "y_label": val_col,
            "labels": rows[label_col].astype(str).tolist(),
            "datasets": [{
                "label": val_col,
                "data": [cls._safe(v) for v in rows[val_col].tolist()],
            }],
        }

    @classmethod
    def _histogram_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        val_col = spec.get("value_column") or (spec.get("value_columns") or [None])[0] or dataframe.columns[0]
        df = dataframe.head(5000)
        series_col = spec.get("series_column")
        section = {
            "kind": "plotly",
            "plot_type": "histogram",
            "title": spec.get("title", "Histogram"),
            "source": str(spec.get("file_id", "data")),
            "row_count": len(df),
            "x_label": val_col,
            "y_label": "Count",
            "bins": spec.get("bins"),
        }
        if series_col and series_col in df.columns:
            section["groups"] = [
                {"name": str(key), "x": [cls._safe(v) for v in group[val_col].tolist()]}
                for key, group in df.groupby(series_col)
            ]
        else:
            section["x"] = [cls._safe(v) for v in df[val_col].tolist()]
        return section

    @classmethod
    def _box_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        val_col = spec.get("value_column") or (spec.get("value_columns") or [None])[0] or dataframe.columns[0]
        df = dataframe.head(5000)
        label_col = spec.get("label_column")
        section = {
            "kind": "plotly",
            "plot_type": "box",
            "title": spec.get("title", "Box Plot"),
            "source": str(spec.get("file_id", "data")),
            "row_count": len(df),
            "x_label": label_col,
            "y_label": val_col,
        }
        if label_col and label_col in df.columns:
            section["groups"] = [
                {"name": str(key), "y": [cls._safe(v) for v in group[val_col].tolist()]}
                for key, group in df.groupby(label_col)
            ]
        else:
            section["y"] = [cls._safe(v) for v in df[val_col].tolist()]
        return section

    @classmethod
    def _heatmap_section(cls, dataframe: pd.DataFrame, spec: dict) -> dict:
        x_col = spec.get("x_column") or dataframe.columns[0]
        y_col = spec.get("y_column") or dataframe.columns[1]
        z_col = spec.get("z_column") or (dataframe.columns[2] if len(dataframe.columns) > 2 else dataframe.columns[-1])
        df = dataframe.head(20000)
        pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
        return {
            "kind": "plotly",
            "plot_type": "heatmap",
            "title": spec.get("title", "Heatmap"),
            "source": str(spec.get("file_id", "data")),
            "row_count": len(df),
            "x_label": x_col,
            "y_label": y_col,
            "z_label": z_col,
            "x": [str(c) for c in pivot.columns.tolist()],
            "y": [str(i) for i in pivot.index.tolist()],
            "z": [[cls._safe(v) for v in row] for row in pivot.values.tolist()],
        }

    @staticmethod
    def _chip(chart_type: str) -> str:
        return {
            "bar": "Bar", "line": "Line", "timeline": "Timeline",
            "scatter3d": "3D Scatter", "surface": "3D Surface",
            "pie": "Pie", "histogram": "Histogram", "box": "Box Plot", "heatmap": "Heatmap",
        }.get(chart_type, chart_type.title() if chart_type else "Chart")

    @classmethod
    def _render_html(cls, title: str, sections: list, source_count: int = 1) -> str:
        chartjs_sections = [s for s in sections if s.get("kind") == "chartjs"]
        plotly_sections = [s for s in sections if s.get("kind") == "plotly"]

        cards = []
        chart_scripts = []
        plot_scripts = []
        chartjs_i = 0
        plotly_i = 0

        for section in sections:
            chip = cls._chip(section.get("chart_type") or section.get("plot_type"))
            if section.get("kind") == "chartjs":
                canvas_id = f"chart{chartjs_i}"
                cards.append(cls._card_html(section, chip, f'<canvas id="{canvas_id}"></canvas>'))
                chart_scripts.append(cls._chartjs_script(canvas_id, section, chartjs_i))
                chartjs_i += 1
            else:
                plot_id = f"plot{plotly_i}"
                cards.append(cls._card_html(section, chip, f'<div id="{plot_id}" class="plot-el"></div>'))
                plot_scripts.append(cls._plotly_script(plot_id, section))
                plotly_i += 1

        plotly_cdn = (
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>'
            if plotly_sections else ""
        )
        generated = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        file_word = "file" if source_count == 1 else "files"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
{plotly_cdn}
<style>{cls._css()}</style>
</head>
<body>
<div class="page">
  <header class="page-header">
    <div>
      <h1>{title}</h1>
      <p class="subtitle">Generated {generated} &middot; {source_count} data {file_word}</p>
    </div>
  </header>
  <main class="dashboard-grid">
    {"".join(cards)}
  </main>
  <footer class="page-footer">Built automatically from your data &mdash; hover any chart to explore.</footer>
</div>
<script>
const PALETTE = {json.dumps(PALETTE)};
function withAlpha(hex, alpha) {{
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}
Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif";
Chart.defaults.font.size = 12.5;
Chart.defaults.color = "#83807A";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.boxHeight = 8;
Chart.defaults.plugins.tooltip.backgroundColor = "#262624";
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.titleFont = {{ weight: '600' }};
{"".join(chart_scripts)}
{"".join(plot_scripts)}
</script>
<script>
(function() {{
  function postHeight() {{
    window.parent.postMessage(
      {{ source: "data-analyzer-chart", height: document.documentElement.scrollHeight }}, "*"
    );
  }}
  window.addEventListener("load", postHeight);
  if (window.ResizeObserver) {{
    new ResizeObserver(postHeight).observe(document.body);
  }} else {{
    window.addEventListener("resize", postHeight);
  }}
  setTimeout(postHeight, 300);
  setTimeout(postHeight, 1000);
}})();
</script>
</body>
</html>
"""

    @staticmethod
    def _card_html(section: dict, chip: str, body_html: str) -> str:
        row_count = section.get("row_count")
        stat = f'<span class="card-stat">{row_count:,} rows</span>' if row_count is not None else ""
        source = section.get("source", "")
        return f"""
    <section class="chart-card">
      <div class="card-header">
        <div>
          <h2>{section["title"]}</h2>
          <p class="card-source">{source}</p>
        </div>
        <div class="card-meta">
          <span class="chip">{chip}</span>
          {stat}
        </div>
      </div>
      <div class="chart-body">{body_html}</div>
    </section>"""

    @staticmethod
    def _chartjs_script(canvas_id: str, section: dict, index: int) -> str:
        chart_type = section["chart_type"]
        datasets = section["datasets"]
        multi = len(datasets) > 1
        datasets_json = json.dumps(datasets)
        is_pie = chart_type in ("pie", "doughnut")

        dataset_styling = f"""
        opts.data.datasets.forEach(function(ds, i) {{
          const color = PALETTE[i % PALETTE.length];
          if ("{chart_type}" === "bar") {{
            ds.backgroundColor = withAlpha(color, 0.82);
            ds.hoverBackgroundColor = color;
            ds.borderRadius = 6;
            ds.borderSkipped = false;
            ds.maxBarThickness = 46;
          }} else if ({json.dumps(is_pie)}) {{
            ds.backgroundColor = ds.data.map(function(_, j) {{ return PALETTE[j % PALETTE.length]; }});
            ds.borderColor = "#FFFFFF";
            ds.borderWidth = 2;
            ds.hoverOffset = 8;
          }} else {{
            ds.borderColor = color;
            ds.backgroundColor = withAlpha(color, 0.12);
            ds.pointBackgroundColor = color;
            ds.pointRadius = {"2.5" if multi else "3"};
            ds.pointHoverRadius = 5;
            ds.borderWidth = 2.5;
            ds.tension = 0.35;
            ds.fill = {"false" if multi else "true"};
          }}
        }});
        """

        x_label = section.get("x_label")
        y_label = section.get("y_label")
        x_title = f'{{ display: true, text: {json.dumps(x_label)}, color: "#A39E92", font: {{ weight: "600" }} }}' if x_label else '{ display: false }'
        y_title = f'{{ display: true, text: {json.dumps(y_label)}, color: "#A39E92", font: {{ weight: "600" }} }}' if y_label else '{ display: false }'
        show_legend = multi or is_pie
        scales_config = "{}" if is_pie else f"""{{
        x: {{ title: {x_title}, grid: {{ display: false }}, ticks: {{ maxRotation: 40, minRotation: 0 }} }},
        y: {{ title: {y_title}, grid: {{ color: "#EDEAE0" }}, beginAtZero: true }}
      }}"""

        return f"""
(function() {{
  const opts = {{
    type: {json.dumps(chart_type)},
    data: {{ labels: {json.dumps(section["labels"])}, datasets: {datasets_json} }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: {json.dumps(show_legend)}, position: 'bottom' }},
      }},
      scales: {scales_config}
    }}
  }};
  {dataset_styling}
  new Chart(document.getElementById({json.dumps(canvas_id)}), opts);
}})();
"""

    @staticmethod
    def _plotly_script(plot_id: str, section: dict) -> str:
        plot_type = section["plot_type"]
        base_layout = {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif", "color": "#3D3929"},
            "margin": {"l": 10, "r": 10, "b": 10, "t": 10},
        }
        cartesian_layout = {
            **base_layout,
            "margin": {"l": 56, "r": 24, "b": 52, "t": 24},
        }
        warm_colorscale = [
            [0.0, "#F5F4EE"], [0.25, "#E8C4A8"], [0.5, "#D9A566"],
            [0.75, "#CC785C"], [1.0, "#8B4A32"],
        ]

        if plot_type in ("surface", "scatter3d"):
            layout = {**base_layout, "scene": {
                "xaxis": {"title": section.get("x_label") or "X", "gridcolor": "#EDEAE0"},
                "yaxis": {"title": section.get("y_label") or "Y", "gridcolor": "#EDEAE0"},
                "zaxis": {"title": section.get("z_label") or "Z", "gridcolor": "#EDEAE0"},
            }}
            if plot_type == "surface":
                traces = [{
                    "type": "surface", "x": section["x"], "y": section["y"], "z": section["z"],
                    "colorscale": warm_colorscale, "showscale": False,
                }]
            else:
                traces = [{
                    "type": "scatter3d", "mode": "markers",
                    "x": section["x"], "y": section["y"], "z": section["z"],
                    "marker": {
                        "size": 4, "color": section["z"], "colorscale": warm_colorscale,
                        "opacity": 0.85, "showscale": False,
                    },
                }]

        elif plot_type == "heatmap":
            layout = {**cartesian_layout,
                "xaxis": {"title": section.get("x_label") or "X", "gridcolor": "#EDEAE0", "automargin": True},
                "yaxis": {"title": section.get("y_label") or "Y", "gridcolor": "#EDEAE0", "automargin": True},
            }
            traces = [{
                "type": "heatmap", "x": section["x"], "y": section["y"], "z": section["z"],
                "colorscale": warm_colorscale,
            }]

        elif plot_type == "histogram":
            groups = section.get("groups")
            layout = {**cartesian_layout,
                "xaxis": {"title": section.get("x_label") or "X", "gridcolor": "#EDEAE0", "automargin": True},
                "yaxis": {"title": section.get("y_label") or "Count", "gridcolor": "#EDEAE0", "automargin": True},
                "barmode": "overlay",
                "showlegend": bool(groups),
                "legend": {"orientation": "h", "y": -0.25},
            }
            if groups:
                traces = []
                for i, group in enumerate(groups):
                    trace = {
                        "type": "histogram", "x": group["x"], "name": group["name"], "opacity": 0.65,
                        "marker": {"color": PALETTE[i % len(PALETTE)]},
                    }
                    if section.get("bins"):
                        trace["nbinsx"] = section["bins"]
                    traces.append(trace)
            else:
                trace = {"type": "histogram", "x": section["x"], "marker": {"color": PALETTE[0]}}
                if section.get("bins"):
                    trace["nbinsx"] = section["bins"]
                traces = [trace]

        elif plot_type == "box":
            groups = section.get("groups")
            layout = {**cartesian_layout,
                "xaxis": {"title": section.get("x_label"), "gridcolor": "#EDEAE0", "automargin": True},
                "yaxis": {"title": section.get("y_label") or "Y", "gridcolor": "#EDEAE0", "automargin": True},
                "showlegend": False,
            }
            if groups:
                traces = [
                    {"type": "box", "y": group["y"], "name": group["name"], "marker": {"color": PALETTE[i % len(PALETTE)]}}
                    for i, group in enumerate(groups)
                ]
            else:
                traces = [{
                    "type": "box", "y": section["y"], "marker": {"color": PALETTE[0]},
                    "name": section.get("y_label") or "",
                }]

        else:
            raise ValueError(f"Unsupported plotly plot_type '{plot_type}'")

        config = {"displaylogo": False, "responsive": True}
        return (
            f"Plotly.newPlot({json.dumps(plot_id)}, {json.dumps(traces)}, "
            f"{json.dumps(layout)}, {json.dumps(config)});"
        )

    @staticmethod
    def _css() -> str:
        return """
:root {
  --bg: #F5F4EE;
  --card-bg: #FFFFFF;
  --border: #E8E4D9;
  --text: #262624;
  --text-muted: #83807A;
  --accent: #CC785C;
  --accent-dark: #B35F45;
  --accent-soft: #F3E4DC;
  --shadow: 0 1px 2px rgba(61,57,41,0.05), 0 4px 16px rgba(61,57,41,0.07);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.page { max-width: 1360px; margin: 0 auto; padding: 40px 32px 64px; }
.page-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 32px; }
.page-header h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.02em; }
.subtitle { margin: 0; color: var(--text-muted); font-size: 14px; }
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
  gap: 24px;
}
.chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 22px 24px 18px;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.chart-card:hover { box-shadow: 0 2px 4px rgba(61,57,41,0.06), 0 12px 28px rgba(61,57,41,0.10); transform: translateY(-1px); }
.card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.card-header h2 { font-size: 15.5px; font-weight: 650; margin: 0 0 2px; letter-spacing: -0.01em; }
.card-source { margin: 0; font-size: 12px; color: var(--text-muted); }
.card-meta { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.chip {
  background: var(--accent-soft); color: var(--accent-dark); font-size: 11px; font-weight: 650;
  padding: 4px 9px; border-radius: 999px; letter-spacing: 0.02em; white-space: nowrap;
}
.card-stat { font-size: 12px; color: var(--text-muted); white-space: nowrap; }
.chart-body { position: relative; height: 340px; }
.plot-el { width: 100%; height: 100%; }
.page-footer { margin-top: 40px; text-align: center; font-size: 12.5px; color: var(--text-muted); }
@media (max-width: 520px) {
  .page { padding: 24px 16px 48px; }
  .dashboard-grid { grid-template-columns: 1fr; }
}
"""
