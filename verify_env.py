import importlib
packages = [
    ("pandas", "pandas"),
    ("duckdb", "duckdb"),
    ("streamlit", "streamlit"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("tensorflow", "tensorflow"),
]
for pkg_name, mod_name in packages:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "version attr missing")
        print(f"{pkg_name.ljust(12)} OK — {ver}")
    except Exception as e:
        print(f"{pkg_name.ljust(12)} FAIL — {e.__class__.__name__}: {e}")
