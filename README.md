# Системи комп'ютерного зору (частина 1) — лабораторні роботи

## Середовище

Python 3.14, Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` — основні бібліотеки з зафіксованими версіями,
`requirements-full.txt` — повний `pip freeze` (разом із Jupyter).

Використовується OpenCV 5.0 (`opencv-contrib-python`); усі функції, згадані в практикумі, у ній присутні.

## Структура

```
labNN/
├── README.md     мета, як запустити, які файли отримаються
├── src/          код
├── data/         малі вхідні дані або посилання на великі
├── results/      числові результати (CSV)
├── figures/      рисунки для звіту
└── report/       звіт у PDF
```

Зображення й відео в репозиторій не потрапляють (див. `.gitignore`), окрім рисунків для звіту в `figures/`.
