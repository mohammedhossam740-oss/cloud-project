import zipfile
import pandas as pd
from src.data import ingest_zip


def test_ingest_real_backblaze_format_stub(tmp_path):
    """Tests streaming layout, not a real Backblaze performance experiment."""
    path = tmp_path / "data_Qx.zip"
    with zipfile.ZipFile(path, "w") as z:
        for day in ("2023-01-01", "2023-01-02"):
            data = f"date,serial_number,model,capacity_bytes,failure,smart_5_raw,smart_194_raw\n{day},AAA,ST12000NM0008,12000000000000,0,2,32\n{day},BBB,SAMSUNG SSD,50000000,0,0,30\n"
            z.writestr(f"{day}.csv", data)
    df = ingest_zip(path, ["ST12000NM0008"])
    assert len(df) == 2
    assert pd.Timestamp("2023-01-02") in set(df.date)
    assert df.smart_5_raw.iloc[0] == 2
