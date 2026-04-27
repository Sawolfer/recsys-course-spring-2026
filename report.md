## Homework 2 Report

### Abstract

Предложен ML-реранкер на основе случайного леса (Random Forest), который переранжирует кандидатов, полученных от SasRec-I2I, используя контентные признаки треков и контекст истории прослушивания пользователя. Модель обучается предсказывать ожидаемое время прослушивания (`listen_time`) на синтетических данных, сгенерированных на основе каталога треков и реалистичных пользовательских предпочтений (жанры, артисты, настроение). На этапе сервинга реранкер получает кандидатов от I2I, вычисляет признаки взаимодействия (жаккардово сходство жанров, совпадение артиста, совпадение настроения, близость года, популярность), и возвращает трек с максимальным предсказанным временем прослушивания. Контроль — SasRec-I2I, тритмент — ML-реранкер.

### Детали реализации

**Архитектура решения:**

```
User Request (POST /next/<user>)
  ├── [Control, 50%]  SasRec-I2I Recommender ──→ track
  └── [Treatment, 50%] ML Reranker
        ├── Get user history (Redis listen_history DB 2)
        ├── Get I2I candidates   (Redis SasRec DB 4)
        ├── Build user context features (genre set, artist set, mood counter, avg year)
        ├── For each candidate: compute 7 interaction features
        │   1. genre_jaccard     — жаккардово сходство жанров
        │   2. artist_in_history — был ли артист в истории
        │   3. same_artist_last  — тот же артист, что и последний трек
        │   4. same_mood_last    — то же настроение, что и у последнего трека
        │   5. dominant_mood_match — совпадение с доминирующим настроением истории
        │   6. year_diff_norm    — нормализованная разница года
        │   7. log_artist_fans   — лог-нормализованная популярность артиста
        ├── RandomForestRegressor.predict() → score
        └── Return candidate with max score (or fallback to Random)
```

**Обучение модели:**
1. `generate_training_data.py` генерирует ~600K примеров на основе `tracks.json`: создаются 4000 синтетических пользователей с предпочтениями по жанрам (top-40), артистам и настроениям; сессии моделируют реалистичное поведение — 55% треков от предпочитаемых артистов, 27% по жанровому совпадению, 10% по настроению, 8% случайных.
2. `train_model.py` обучает `RandomForestRegressor` (50 деревьев, max_depth=12, min_samples_leaf=10, random_state=42). Итоговый R² на кросс-валидации: ~0.14. Наиболее важные признаки: `artist_in_history` (60%), `same_artist_last` (28%), `genre_jaccard` (6%).
3. Модель сохраняется в `botify/data/ml_model.joblib`, признаки треков — в `botify/data/track_features.pkl`.

**Файлы:**
- `botify/botify/recommenders/ml_reranker.py` — класс MLReranker
- `botify/data/generate_training_data.py` — генерация обучающих данных
- `botify/data/train_model.py` — обучение модели
- `botify/botify/experiment.py` — добавлен эксперимент `ML_RERANKER`
- `botify/botify/server.py` — роутинг тритмента на MLReranker

### Результаты A/B эксперимента

| Метрика | Контроль (C) | Тритмент (T1) | Effect, % | CI lower, % | CI upper, % | Значимо |
|---------|-------------|---------------|-----------|-------------|-------------|---------|
| mean_time_per_session | * | * | * | * | * | * |
| mean_tracks_per_session | * | * | * | * | * | * |
| mean_request_latency | * | * | * | * | * | * |

> Результаты будут заполнены после прогона симулятора (`make run`). Данные берутся из `ab_result.json`.
