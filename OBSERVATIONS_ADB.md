
# 1. Relative Import
With the following project structure, without using any `__init__.py`, the relative import works for
- main.py import from all of `src/` modules and packages
- script.py import from `utils.py` (same directory) and `text_processing` package (child directory)
```
.
├── src/
│   ├── text_processing/
│   │   └── text_processor.py
│   ├── utils.py
│   └── script.py
└── main.py
```