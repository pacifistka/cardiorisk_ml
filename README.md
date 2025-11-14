# Модель для предсказания риска сердечного приступа

Проект: сервис на FastAPI, который по данным пациента предсказывает риск сердечного приступа.

## Содержание проекта:

- Jupyter Notebook `notebooks/cardiorisk.ipynb`:
  - исследование данных;
  - предобработка (удаление технических и "утекающих" признаков);
  - изучение статистик данных и офомление графиков;
  - обучение моделей и подбор порога вероятности;
  - сохранение артефактов модели и файла с предсказаниями.

- FastAPI-приложение `src/app.py`:
  - загружает обученную модель;
  - принимает путь к CSV с тестовой выборкой;
  - возвращает предсказания в формате JSON.

Модель и порог хранятся в `artifacts/pipeline.joblib` и `artifacts/meta.json`.

Файл `data/heart_predictions.csv` — итоговые предсказания на тестовой выборке (2 колонки: `id`, `prediction`), он используется для проверки качества модели скриптом `test.py`.

## Данные

Учебные датасеты (`heart_train.csv`, `heart_test.csv`) **не входят в репозиторий**.
(Локально в папке `data/` лежат: `data/heart_train.csv` и`data/heart_test.csv`)

При наличии технических столбцов (например, `Unnamed: 0`) сервис их автоматически игнорирует: модель обучена без этих признаков.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Запуск API
python -m uvicorn src.app:app --reload

Документация будет доступна по адресу:

Swagger UI: http://127.0.0.1:8000/docs 


Пример запроса
POST /predict

{
  "csv_path": "data/heart_test.csv"
}


Пример ответа:

{
  "n_samples": 321,
  "predictions": [
    {"id": 0, "prediction": 0},
    {"id": 1, "prediction": 1}
  ]
}
