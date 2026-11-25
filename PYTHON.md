# PycRune bằng Python

Bản chuyển ngữ của PycRune từ commit `95497dc86091b5bb4c9fe18f428adc92374ec95c`. Tác giả dự án gốc là Vương Tiến Đạt (Vuong Tien Dat, 王达). [Kho nguồn gốc](https://github.com/hngocuyen/pycrune/tree/95497dc86091b5bb4c9fe18f428adc92374ec95c)

Decompiler dựng mã Python từ lệnh bytecode, luồng điều khiển, mô phỏng ngăn xếp và bảng ngoại lệ. Gói có đủ năm lệnh của dự án gốc và API Python để nhúng vào chương trình khác. Mã chạy bằng thư viện chuẩn của CPython.

## Chạy từ mã nguồn

Mở terminal ở thư mục gốc của repo. Dùng CPython 3.8–3.15.

```bash
python pycrune.py --help
python pycrune.py decompile input.pyc -o output.py
```

Nếu muốn cài lệnh `pycrune` vào môi trường Python hiện tại, chạy `python -m pip install .`. Bạn cũng có thể dùng `python -m pycrune` thay cho `python pycrune.py`.

## Lệnh

| Lệnh sau `python pycrune.py` | Chức năng |
| --- | --- |
| `decompile input.pyc -o output.py` | Khôi phục mã và lưu ra tệp. Có thể dùng đầu vào `.py` để biên dịch rồi dịch ngược. |
| `decompile input.pyc` | In mã đầy đủ ra stdout. |
| `input.pyc` | Lưu cạnh tệp đầu vào dưới tên `dcp<X.Y>-<tên>.py`. |
| `compare input.pyc output.py` | Biên dịch lại và đối chiếu bytecode của toàn bộ code object, kể cả hàm và lớp lồng nhau. |
| `verify input.pyc output.py` | Đối chiếu bytecode và chạy cả hai chương trình để so stdout và mã thoát. |
| `batch folder` | Kiểm tra dịch ngược mọi `.pyc` trong cây thư mục. |
| `batch folder -o recovered` | Đồng thời ghi kết quả theo cấu trúc thư mục tương đối. |
| `inspect input.pyc --instructions` | Xem các khối CFG, vùng xử lý ngoại lệ, vòng lặp và lệnh bytecode. |

`decompile` và `batch` mặc định chỉ xuất kết quả khi bytecode khớp sau khi biên dịch lại. Thêm `--unchecked` để lấy bản dịch ngược khi cần đọc hoặc chỉnh sửa. Cú pháp của đầu ra vẫn được kiểm tra. `decompile --verify` và `batch --verify` còn thực thi chương trình gốc và bản dựng lại, nên dùng với mã bạn tin cậy.

Với tệp `.pyc`, chương trình đọc magic number để chọn CPython. Đặt bản tương ứng trên PATH dưới tên như `python3.11`, dùng Python Launcher trên Windows hoặc chỉ định `--python`.

```bash
python pycrune.py decompile input.pyc --python /path/to/python3.11 -o output.py
python pycrune.py decompile input.opt-2.pyc --optimize 2 -o output.py
```

Với tệp bị bỏ header, hãy dùng CPython đã tạo tệp hoặc chỉ định `--python`. Đầu vào `.py` dùng interpreter hiện tại trừ khi có `--python`. Trên terminal tương tác, lệnh dịch một tệp hiện phần trăm bên phải thanh tiến trình và xoá thanh khi xong.

## API Python

```python
from pycrune import compare_file, decompile_file, load_bytecode

source = decompile_file("input.pyc")
result = compare_file("input.pyc", source)
print(result.is_identical)
print(result.discrepancies)

model = load_bytecode("input.pyc")
print(model.python_version)
```

`decompile_code(code)` nhận code object của interpreter hiện tại hoặc mô hình do `load_bytecode` trả về. Dùng `with PyBridge.for_path(path) as bridge` khi cần đọc hoặc biên dịch nhiều tệp cùng bản CPython trong một tiến trình con. Các API dựng mã trả về văn bản để bạn tự chọn cách sử dụng. `compare_file` xác nhận kết quả bằng interpreter tương ứng.

## Cấu trúc

| Mô-đun | Vai trò |
| --- | --- |
| `interpreter.py`, `bridge.py`, `bytecode.py` | Nhận diện phiên bản, nạp marshal và chuẩn hóa opcode. |
| `model.py`, `nodes.py`, `stack.py` | Mô hình bytecode, cây cú pháp và ngăn xếp ký hiệu. |
| `cfg.py` | Phân tích luồng điều khiển, dominator, vùng ngoại lệ và vòng lặp. |
| `structural.py`, `cleanup.py`, `emitter.py` | Dựng cây cú pháp, dọn cấu trúc do compiler tạo và sinh mã Python. |
| `verification.py`, `cli.py` | So sánh, kiểm tra khi chạy và các lệnh terminal. |

## Kiểm tra

Chạy `python -m unittest discover -s tests`. Trên Linux, 215/225 lượt đối chiếu bytecode khớp trên tám phiên bản CPython từ 3.8 đến 3.15. Các lệnh được thử thêm với tệp tạo từ phiên bản khác interpreter đang chạy. Một số dạng bytecode phức tạp vẫn còn giới hạn, gồm lệnh `with` có nhiều context manager và `return` trên 3.8–3.10, một số nhánh `try/finally` và template string `t"…"`. Chế độ kiểm tra mặc định báo lỗi cho các kết quả lệch bytecode.
