from pathlib import Path

from dproblem.compat import import_openpyxl_load_workbook

load_workbook = import_openpyxl_load_workbook()


def read_sheet_rows(path, sheet_name="数据"):
    workbook = load_workbook(
        filename=Path(path), read_only=True, data_only=True
    )
    try:
        sheet = workbook[sheet_name]
        return [list(row) for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()


def extract_table(rows, first_header, second_header=None):
    """Extract a contiguous table located by its header text.

    This avoids depending on cosmetic blank rows or fixed Excel row numbers.
    """

    header_index = None
    for index, row in enumerate(rows):
        if not row or row[0] != first_header:
            continue
        if second_header is not None and (len(row) < 2 or row[1] != second_header):
            continue
        header_index = index
        break
    if header_index is None:
        raise ValueError("table header not found: {!r}".format(first_header))

    headers = rows[header_index]
    last_nonempty = max(i for i, value in enumerate(headers) if value is not None)
    headers = [str(value) for value in headers[: last_nonempty + 1]]
    records = []
    for row in rows[header_index + 1 :]:
        if not row or row[0] is None:
            break
        values = list(row[: len(headers)])
        values.extend([None] * (len(headers) - len(values)))
        records.append(dict(zip(headers, values)))
    return records
