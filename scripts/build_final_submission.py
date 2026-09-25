"""Populate the official workbook with the frozen Q1-Q4 solution tables."""

import argparse
import csv
import json
import sys
from copy import copy
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def coerce(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def copy_row_style(sheet, source_row, target_row, width):
    sheet.row_dimensions[target_row].height = sheet.row_dimensions[source_row].height
    for column in range(1, width + 1):
        source = sheet.cell(source_row, column)
        target = sheet.cell(target_row, column)
        if source.has_style:
            target._style = copy(source._style)
        target.font = copy(source.font)
        target.fill = copy(source.fill)
        target.border = copy(source.border)
        target.alignment = copy(source.alignment)
        target.protection = copy(source.protection)
        target.number_format = source.number_format


def fill_sheet(sheet, rows):
    headers = [sheet.cell(1, column).value for column in range(1, sheet.max_column + 1)]
    headers = [header for header in headers if header is not None]
    width = len(headers)
    previous_max_row = sheet.max_row
    for row_index in range(2, max(previous_max_row, len(rows) + 1) + 1):
        for column in range(1, width + 1):
            sheet.cell(row_index, column).value = None
    for row_index, row in enumerate(rows, start=2):
        if row_index > previous_max_row:
            copy_row_style(sheet, 2, row_index, width)
        for column, header in enumerate(headers, start=1):
            sheet.cell(row_index, column).value = coerce(row.get(header))
    sheet.auto_filter.ref = "A1:{}{}".format(
        sheet.cell(1, width).column_letter, len(rows) + 1
    )
    sheet.freeze_panes = "A2"


def build_submission(project_root, output_path):
    root = Path(project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.compat import import_openpyxl_load_workbook
    from dproblem.q4.partition import verify_s3_freeze

    verify_s3_freeze(root)
    q4_summary = json.loads((root / "results/q4/q4_summary.json").read_text(encoding="utf-8"))
    if q4_summary["status"] != "PASS":
        raise ValueError("Q4 is not PASS")

    tables = {
        "Q1_单点组批": read_csv(root / "results/q1/main_trips.csv"),
        # Q3 locally adjusted the Q2 parent schedule.  The final workbook uses
        # that coherent joint transport schedule so Q2 and Q3 times agree.
        "Q2_运输架次": read_csv(root / "results/q3/transport_trips.csv"),
        "Q2_逐箱交付": read_csv(root / "results/q3/transport_box_deliveries.csv"),
        "Q3_中继架次": read_csv(root / "results/q3/relay_sorties.csv"),
        "Q3_通信保障": read_csv(root / "results/q3/communication_guarantee.csv"),
        "Q4_分区配置": read_csv(root / "results/q4/selected_partition.csv"),
    }
    load_workbook = import_openpyxl_load_workbook()
    workbook = load_workbook(root / "结果提交模板.xlsx")
    try:
        for sheet_name, rows in tables.items():
            fill_sheet(workbook[sheet_name], rows)
        # Preserve exact volume visibility in the final workbook.
        q1 = workbook["Q1_单点组批"]
        for row in range(2, len(tables["Q1_单点组批"]) + 2):
            q1.cell(row, 6).number_format = "0.000000"
        for sheet_name in ("Q1_单点组批", "Q2_运输架次", "Q2_逐箱交付", "Q3_中继架次", "Q3_通信保障"):
            sheet = workbook[sheet_name]
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"
        workbook.calculation.fullCalcOnLoad = True
        workbook.calculation.forceFullCalc = True
        output = Path(output_path)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(output)
    finally:
        workbook.close()
    return output, {name: len(rows) for name, rows in tables.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="results/结果提交_Q1-Q4.xlsx")
    args = parser.parse_args()
    output, counts = build_submission(args.project_root, args.output)
    print(json.dumps({"output": str(output), "row_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
