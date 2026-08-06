# راه‌اندازی نسخه آزمایشی داشبورد گذران وقت

## 1) ساخت محیط مجازی در PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

اگر PowerShell اجازه فعال‌سازی نداد، ابتدا این دستور را در همان پنجره اجرا کنید:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2) آزمایش Streamlit

```powershell
python -m streamlit run app_test.py
```

## 3) قرار دادن داده

فایل زیر را در پوشه `data/raw` کپی کنید:

`TimeUse_Cleaned_AllWaves(2).dta`

## 4) ساخت فایل سبک داشبورد

```powershell
python prepare_data.py
```

## 5) اجرای داشبورد

```powershell
python -m streamlit run app.py
```

برای توقف برنامه در ترمینال `Ctrl+C` بزنید.
