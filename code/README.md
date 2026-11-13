# Код вычислительного эксперимента

**Вопрос.** Сколько попарных сравнений носителей нужно, чтобы правильно упорядочить
кандидатов на замену русского заимствования, если в модель Брэдли–Терри добавить
лингвистические признаки слова?

**Модель.** $P(w_i \succ w_j) = \sigma(u_i - u_j)$, $u_i = \theta^\top x_i + \eta_i$,
$\eta_i \sim \mathcal N(0, \lambda^{-1})$ — иерархическая BT. Базовые модели:
BT без признаков ($u_i = \eta_i$) и модель только по признакам ($u_i = \theta^\top x_i$).
Регуляризация $\lambda$ для BT и иерархической модели выбирается по одной сетке
на отложенных 20 % сравнений.

**Синтетические данные.** 40 заимствований × 8 кандидатов, 5 признаков;
$\rho = \mathrm{Var}(\theta^\top x)/\mathrm{Var}(u)$ задаёт, какую долю предпочтений
объясняют признаки. Критерий — $\tau$ Кендалла внутри заимствования между
оценённым и истинным порядком, 20 запусков, 95 %-е бутстреп-интервалы.

## Запуск

```bash
pip install -r requirements.txt
python code/main.py      # все вычисления, ~1.5 мин на 2 ядрах -> results/
python code/plots.py     # рисунки из сохранённых результатов -> figures/
python code/tests/test_models.py   # юнит-тесты (или python -m pytest code/tests)
```

`python code/main.py --quick` — короткий прогон за несколько секунд (в `results/quick/`).

## Структура

| файл | что делает |
|---|---|
| `config.py` | все параметры эксперимента и описание его хода |
| `data.py` | синтетический лексикон и моделирование сравнений носителей |
| `models.py` | три модели (MAP-оценки BT) и выбор $\lambda$ на отложенной выборке |
| `metrics.py` | $\tau$ Кендалла и расчёт экономии сравнений $m_{BT}/m_0$ |
| `main.py` | единственная точка запуска вычислений, сохраняет `results/*.csv` |
| `plots.py` | только читает `results/` и строит `figures/main_figure.{pdf,png}` (для статьи) и `figures/main_figure_slide.{pdf,png}` (компактная версия для слайда) |

Результаты детерминированы: повторный запуск даёт побитово те же `results/*.csv`.

## Дополнения после первого отчёта (13.11)

```bash
pip install -r requirements.txt          # + wordfreq, pyphen (E4)
python code/main.py                      # всё: E1, E2, E3, E4, E5, E7, таблица E6 (E8 пропускается без кэша LLM)
python code/main.py --only synthetic     # E1 (+ дополнения) и E2
python code/main.py --only designs       # E3 (план random берётся из E2)
python code/main.py --only real          # E4 и E5 (скачивает BenchLS и WCL в data/raw/ с проверкой sha256)
python code/main.py --only runtime       # E7
python code/main.py --quick              # короткий прогон всех частей -> results/quick/, tables/quick/
python code/plots.py                     # все рисунки из results/ (не пересчитывает)
python -m pytest code/tests -q           # тесты
```

| эксп. | что считает | файлы в `results/` |
|---|---|---|
| E1 | исходный E1 без изменений (`raw_results.csv`, `curves.csv`, `savings.csv`, `summary.json`); в том же прогоне — `bt_fixed` (λ = 1/V), `hier_oracle` (известное μ, λ = 1/((1−ρ)V)), MSE контрастов, top-1, NLL на обучении и на 20 000 свежих контрольных сравнений | `e1_raw.csv`, `e1_curves.csv`, `e1_savings.csv` |
| E2 | отдельный прогон (бюджеты 0.25…128, 30 семян, свои потоки): измеренная экономия по τ и MSE (иерархическая против BT, оракулы) и все предсказания теории: замкнутая форма, спектральное решение, границы, `expected_risk_mc` (mean_field / design / full), разложение расхождения | `theory_raw.csv`, `theory_curves.csv`, `theory_mc_risk.csv`, `theory_check.csv` |
| E3 | планы опроса random / cycle / star, ρ ∈ {0.3, 0.5, 0.7}; измерение и предсказания для каждого плана | `designs_raw.csv`, `designs_mc_risk.csv`, `designs_savings.csv` |
| E4 | реальные данные: 10 разбиений аннотаторов 6/5, бюджеты 0.25…24, `bt` и для 6 наборов признаков ({charLM}, {слоги}, {длина}, {zipf}, {zipf, длина}, все 4) `features` и `hierarchical`; λ подбирается **по отложенному аннотатору** группы A; τ, top-1, NLL; согласие групп A и B; экономия при m₀ ∈ {0.25…4}; теория (ρ̂, V̂, Ī_e по û; среднее поле на всех разбиениях, аналог `full` по 100 подвыборкам на разбиении 0, float32) | `real_raw.csv`, `real_curves.csv`, `real_savings.csv`, `real_theory.csv`, `real_full_risk.csv`, `real_dataset.json` |
| E5 | сложность модели (первые d признаков, m ∈ {2, 8}, λ по аннотатору) и путь по λ (m = 2; NLL отложенного аннотатора и прежних 20 % сравнений) | `real_complexity.csv`, `real_lambda_path.csv` |
| E6 | таблица сравнения моделей (только из сохранённых результатов) | `tables/model_comparison.{csv,tex}` |
| — | отношение «измеренный MSE / риск full» для `bt_fixed`, `hier_oracle` по планам (из сохранённых E3, без пересчёта) | `designs_mse_vs_risk.csv` |
| E7 | время подгонки от G (последовательно, BLAS в один поток) и наклон в log–log | `runtime.csv`, `runtime_fit.csv` |

Новые модули: `theory.py` (лапласово приближение и `expected_risk_mc`), `real_data.py` (данные и признаки),
`real_exp.py` (задачи E4/E5), `synthetic.py` (задачи E2/E3/E7), `experiments.py` (оркестровка E1+…E8),
`aggregate.py` (бутстреп-сводки). Рисунки: `fig_theory_check`, `fig_designs`, `fig_real`, `fig_real_theory`, `fig_complexity`;
`python code/plots.py --only fig_real fig_real_theory` перерисовывает только названные.

Детерминированность: у каждого нового источника случайности свой поток `default_rng([...])`
(см. докстринги `synthetic.py`, `real_exp.py`); старые потоки E1 не тронуты. Все CSV пишутся с LF.
Время работы — в `results/timing.json` (не в CSV).

Время (Windows, 6 логических ядер): E1 ≈ 1.5 мин, E2 ≈ 3.5 мин, E3 ≈ 5 мин, E4 ≈ 30 мин (из них ≈ 15 мин —
аналог `full` для 6 наборов признаков), E5 ≈ 1 мин, E7 ≈ 3 с; полный `main.py` ≈ 41 мин; `--quick` < 1 мин.
