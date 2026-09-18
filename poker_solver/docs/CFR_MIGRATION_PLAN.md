# План миграции: Fictitious Play → Counterfactual Regret Minimization (CFR)

Документ фиксирует контекст и план перехода с Fictitious Play (FP) на CFR в проекте `poker_solver`. Реализация **не запланирована сейчас** — текущий FP достаточен для практики.

---

## 1. Зачем CFR

### Проблема FP

Fictitious Play сходится медленно. Единственное значимое отклонение от HRC в 3-max:

| Диапазон | Наш FP | HRC | Δ |
|----------|--------|-----|---|
| SB push (BTN fold) | 59.4% | 63.8% | **−4.4%** |

Остальные 5 из 6 диапазонов — в пределах ±3% от HRC.

### Преимущества CFR

- **Скорость сходимости:** CFR сходится в **5–10× быстрее**, чем FP, и даёт точность **±1%** за то же время.
- **Индустриальный стандарт:** PioSOLVER, HRC, GTO+ и другие покерные солверы используют CFR или его вариации (CFR+, DCFR, MCCFR).
- **Теоретическая обоснованность:** CFR минимизирует counterfactual regret и даёт стратегию, математически ближе к Nash equilibrium, чем FP.

---

## 2. Текущее состояние

| Компонент | Расположение | Описание |
|-----------|--------------|----------|
| HU солвер | `core/src/solver.rs` (`solve_hu`) | Heads-up push/fold |
| 3-max солвер | `core/src/three_max.rs` (`solve_3max`) | Три игрока, push/fold |
| Алгоритм | Fictitious Play | Итеративное усреднение стратегий |
| Точность | 5/6 диапазонов | ±3% от HRC |

### Связанные модули

- `core/src/equity.rs` — HU equity
- `core/src/equity_3way.rs` — 3-way showdown equity
- `core/src/equity_cache.rs` — кэш equity для 169 комбо
- `core/src/icm.rs` — ICM-расчёты
- `core/src/bin/solve.rs` — CLI

---

## 3. План миграции

### Этап A: Абстракция (2–3M токенов)

Выделить trait `SolverAlgorithm` с методами:

```rust
trait SolverAlgorithm {
    fn init(ranges: &Ranges) -> State;
    fn iterate(state: &mut State) -> ();
    fn converged(state: &State) -> bool;
    fn strategy(state: &State) -> Strategy;
}
```

- Реализовать `FictitiousPlay` как одну из реализаций trait.
- Перенести текущую логику FP из `solver.rs` / `three_max.rs` в `fp.rs`.
- **Проверка:** результаты FP не изменились (регрессионный тест против текущих значений).

### Этап B: CFR для HU (3–5M токенов)

- Реализовать `CfrState` для HU в `core/src/cfr.rs`.
- Каждому действию (push / fold / call) сопоставить regret-вектор длины 169 (по комбо).
- На каждой итерации:
  1. Посчитать counterfactual value каждого действия.
  2. Обновить regret: `regret[a] += cf_value[a] - cf_value[strategy]`.
  3. Построить стратегию через regret-matching: `σ(a) = max(0, regret[a]) / Σ max(0, regret)`.
- Сравнить с HRC: ожидание **±1%**.

### Этап C: CFR для 3-max (3–5M токенов)

- Расширить `CfrState` на 3 игроков (BTN, SB, BB).
- Учесть 3-way showdown через `equity_3way`.
- Counterfactual regret для каждого игрока на каждом decision point.
- Сравнить с HRC: ожидание **±1.5%** по всем 6 диапазонам.

### Этап D: Опциональные улучшения

| Вариант | Описание | Когда применять |
|---------|----------|-----------------|
| **CFR+** | Обнуление отрицательных regret после каждой итерации | Если базовый CFR сходится медленно |
| **Discounted CFR (DCFR)** | Экспоненциальное забывание старых итераций | Для ускорения на поздних итерациях |
| **Vector-form CFR** | Параллельный расчёт по всем 169 комбо | Для многоядерных CPU |

---

## 4. Архитектурные решения

- **Fallback:** существующий FP-код остаётся как `FictitiousPlay` — не удалять.
- **Модуль:** CFR в отдельном файле `core/src/cfr.rs`.
- **Выбор алгоритма:** флаг CLI `--algorithm fp|cfr` (по умолчанию `fp`).
- **API:** не ломать существующий `SolverInput` / `SolverOutput`.
- **Тесты:** для каждого этапа — сравнение с HRC-референсами (зашить ожидаемые диапазоны в тесты).

### Целевая структура

```
core/src/
├── solver.rs          # публичный API (solve, SolverInput, SolverOutput)
├── three_max.rs       # 3-max game tree
├── algorithm/
│   ├── mod.rs         # trait SolverAlgorithm
│   ├── fp.rs          # FictitiousPlay
│   └── cfr.rs         # CounterfactualRegretMinimization
├── equity.rs
├── equity_3way.rs
└── ...
```

---

## 5. Риски

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| CFR даёт другой результат, не обязательно «лучше» | Средняя | HRC использует свою вариацию CFR; точное совпадение ±1% может не получиться — зафиксировать допустимый порог |
| 3-way CFR сложнее: counterfactual regret для трёх игроков одновременно | Высокая | Начать с HU, перенести паттерны в 3-max |
| Производительность: больше памяти на regret | Низкая | 169 × 3 действия × 3 игрока = **1521** значений на итерацию — не критично |
| Регрессия FP при рефакторинге (Этап A) | Средняя | Регрессионные тесты до и после абстракции |

---

## 6. Когда делать

- **Не сейчас.** Текущий FP работает достаточно для практики (5/6 диапазонов в ±3%).
- **Триггеры для старта:**
  - Нужна точность **±1%** по всем диапазонам.
  - Добавление **raise/call** (не только push/fold) — FP плохо масштабируется на много действий.
  - Появление новых форматов (4-max, 6-max), где FP не успевает сойтись за разумное время.
- **Приоритет:** средний. **9-max важнее.**

---

## 7. Референсы

### Статьи

1. Neller, T. & Lanctot, M. (2013). *An Introduction to Counterfactual Regret Minimization*. [PDF](https://www.cs.unr.edu/~tyler/papers/cfr-tutorial.pdf)
2. Zinkevich, M. et al. (2007). *Regret Minimization in Games with Incomplete Information*. NIPS 2007.

### Open-source реализации

- [TinkeringIdiot/CFR-poker](https://github.com/TinkeringIdiot/CFR-poker) — CFR для покера на C++
- [DrStug/poker-solver](https://github.com/DrStug/poker-solver) — Python, push/fold HU

### Внутренние референсы (HRC)

Зафиксировать текущие значения для регрессионных тестов после миграции:

| Диапазон | HRC | FP (текущий) |
|----------|-----|--------------|
| BTN push | — | — |
| SB push (BTN fold) | 63.8% | 59.4% |
| BB call vs BTN | — | — |
| BB call vs SB | — | — |
| SB push (BTN call) | — | — |
| BTN fold | — | — |

> Заполнить все 6 диапазонов актуальными значениями перед началом Этапа B.
