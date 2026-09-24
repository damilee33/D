import json
from pathlib import Path

from dproblem.io.excel import extract_table, read_sheet_rows


def load_paths(project_root):
    path = Path(project_root) / "config" / "paths.json"
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    return {name: Path(project_root) / value for name, value in values.items()}


def load_nodes(path):
    rows = read_sheet_rows(path)
    centers = extract_table(rows, "调度中心编号", "调度中心名称")
    service_areas = extract_table(rows, "服务区编号", "服务区名称")
    return {"centers": centers, "service_areas": service_areas}


def load_demands(path):
    summary_rows = read_sheet_rows(path, "数据")
    box_rows = read_sheet_rows(path, "逐箱货箱清单")
    return {
        "summary": extract_table(summary_rows, "服务区编号", "物资类型"),
        "boxes": extract_table(box_rows, "货箱编号", "服务区编号"),
    }


def load_transport(path):
    rows = read_sheet_rows(path)
    return {
        "types": extract_table(rows, "机型编号", "机型名称"),
        "drones": extract_table(rows, "无人机编号", "机型编号"),
        "batteries": extract_table(rows, "机型编号", "共享电池组总数（组）"),
    }


def load_relay(path):
    rows = read_sheet_rows(path)
    return {
        "types": extract_table(rows, "机型编号", "机型名称"),
        "drones": extract_table(rows, "中继无人机编号", "机型编号"),
        "modules": extract_table(rows, "机型编号", "共享能源组件总数（组）"),
    }


def load_communication(path):
    rows = read_sheet_rows(path)
    return extract_table(rows, "参数类别", "参数名称")


def load_project_data(project_root):
    paths = load_paths(project_root)
    return {
        "paths": paths,
        "nodes": load_nodes(paths["nodes_workbook"]),
        "demands": load_demands(paths["demand_workbook"]),
        "transport": load_transport(paths["transport_workbook"]),
        "relay": load_relay(paths["relay_workbook"]),
        "communication": load_communication(paths["communication_workbook"]),
    }

