import pandas as pd

def extract_xlsx_chunks(filepath):
    chunks = []
    xls = pd.ExcelFile(filepath)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue
        csv_text = df.to_csv(index=False, encoding="utf-8")
        chunks.append({"type": "sheet", "content": csv_text})
    return chunks

if __name__ == "__main__":
    print(extract_xlsx_chunks("샘플.xlsx"))
