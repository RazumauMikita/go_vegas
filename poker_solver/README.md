# Poker Solver

Локальный инструмент для анализа push/fold в SNG. Только для личного использования вне игры.

## Этап 1

- Представление карт (`Card`)
- Оценщик 7-карточных рук (`HandRank`)
- Тесты evaluator

## Сборка и запуск

Все команды — из корня проекта (`poker_solver/`). Нужен [Rust](https://rustup.rs/) (`cargo`).

### Сборка

```bash
# Debug (быстрее собирается)
cargo build

# Release (нужен для солвера и GUI)
cargo build --release

# Только CLI-солвер
cargo build --release --bin solve

# Только GUI
cargo build --release -p poker_ui
```

Бинарники: `target/release/solve.exe`, `target/release/poker_ui.exe` (на Linux/macOS без `.exe`).

### Запуск GUI

```bash
cargo run --release -p poker_ui
```

Или после сборки: `target/release/poker_ui.exe`.

### Запуск солвера (CLI)

Нужен `equity_cache.bin` в текущей директории (или путь через `--cache`).

```bash
# HU 10bb
cargo run --release --bin solve -- --stacks 1000,1000 --payouts 0.625,0.375 --blinds 50,100 --cache equity_cache.bin

# 3-max 10bb
cargo run --release --bin solve -- --stacks 1000,1000,1000 --payouts 0.5,0.3,0.2 --blinds 50,100 --cache equity_cache.bin
```

Первый 3-max прогон создаёт `equity_3way_cache.bin` (~19 MB) рядом с командой. Повторный запуск читает этот файл и проходит заметно быстрее.

### Тесты

```bash
cargo test
cargo test --release
```

---

## CLI — быстрый справочник

Все команды запускаются из корня проекта (`poker_solver/`).
Флаги после `--` передаются самой программе.

### eval

Оценка 7 карт: категория + HandRank.

```bash
# Royal flush
cargo run --release --bin eval -- "As Ks Qs Js Ts 2c 3d"

# Wheel straight (A-2-3-4-5)
cargo run --release --bin eval -- "Ac 2d 3h 4s 5c Kd Qh"

# Four of a kind
cargo run --release --bin eval -- "Ah Ad Ac As Kd Qc 2h"

# Full house
cargo run --release --bin eval -- "Ah Ad Ac Kd Kc 2h 3s"
```

### equity

Эквити hero vs villain. Две конкретные руки — exact enumeration, диапазоны — Monte Carlo.

```bash
# Preflop AA vs KK
cargo run --release --bin equity -- "Ah Ad" "Kh Kd"

# Флоп
cargo run --release --bin equity -- "Ah Ad" "Kh Kd" --board Ts 7d 2c

# Ривер
cargo run --release --bin equity -- "Ah Ad" "Kh Kd" --board Ts 7d 2c 9h 4s
```

### icm

$EV по модели Malmuth-Harville. `--payouts` должен суммироваться в `1.0`.

```bash
# HU
cargo run --release --bin icm -- --stacks 1000,1000 --payouts 0.65,0.35

# 3-max
cargo run --release --bin icm -- --stacks 5000,3000,2000 --payouts 0.5,0.3,0.2

# Бабл (3 игрока, платят двое)
cargo run --release --bin icm -- --stacks 5000,3000,2000 --payouts 0.65,0.35

# WTA
cargo run --release --bin icm -- --stacks 5000,3000,2000 --payouts 1.0
```

### verify_cache

Проверка `equity_cache.bin`: AA vs KK ≈ 0.81.

```bash
cargo run --release --bin verify_cache -- equity_cache.bin
```

### solve

Nash push/fold. Нужен `equity_cache.bin` в текущей директории (или путь через `--cache`).

```bash
# HU 10bb, ICM leftover 50/30 → 0.625/0.375
cargo run --release --bin solve -- --stacks 1000,1000 --payouts 0.625,0.375 --blinds 50,100 --cache equity_cache.bin

# HU 5bb
cargo run --release --bin solve -- --stacks 500,500 --payouts 0.625,0.375 --blinds 50,100 --cache equity_cache.bin

# HU 20bb
cargo run --release --bin solve -- --stacks 2000,2000 --payouts 0.625,0.375 --blinds 50,100 --cache equity_cache.bin

# HU 10bb WTA
cargo run --release --bin solve -- --stacks 1000,1000 --payouts 1.0 --blinds 50,100 --cache equity_cache.bin

# HU 10bb, payouts 50/30 (солвер нормализует сумму)
cargo run --release --bin solve -- --stacks 1000,1000 --payouts 0.5,0.3 --blinds 50,100 --cache equity_cache.bin

# 3-max 10bb
cargo run --release --bin solve -- --stacks 1000,1000,1000 --payouts 0.5,0.3,0.2 --blinds 50,100 --cache equity_cache.bin
```

### build_cache

Генерация preflop-кэша 169×169. Полный прогон долгий; `--sample` оценивает ETA.

```bash
# Полная генерация
cargo run --release --bin build_cache -- --iterations 100000 --output equity_cache.bin

# Sample (оценка скорости)
cargo run --release --bin build_cache -- --sample 20 --iterations 100000
```

### Эталонные значения

```bash
# HU 10bb (combo-share): SB push 59.2%, BB call 37.8%
# HRC:                   SB push 58.3%, BB call 37.4%

# 3-max 10bb (combo-share):
#   BTN push                26.0%
#   SB call vs BTN           7.1%
#   BB call (SB fold)        8.8%
#   BB call (SB call)        1.4%
#   SB push (BTN fold)      63.8%
#   BB call vs SB           23.1%
```

### Частые проблемы

```bash
# LNK1104: cannot open file ... poker_core-....exe
# Файл в target/debug занят зависшим процессом теста.
# Закрой .exe в Task Manager или перезапусти терминал, затем:
cargo test

# Медленные тесты (equity exact / полная генерация кэша) помечены #[ignore].
# Быстрый прогон без них:
cargo test
# Игнорируемые — только в release:
cargo test --release -- --ignored

# Нет кэша: solve и часть тестов требуют equity_cache.bin рядом с crate.
# Сгенерируй или положи файл в poker_solver/equity_cache.bin, затем:
cargo run --release --bin verify_cache -- equity_cache.bin
```

### Переменные для быстрой замены

`--stacks` — стеки в фишках через запятую. HU: два значения, 3-max: три. 10bb при блайндах 50/100 = `1000,1000`.

`--payouts` — доли призовых. Для `icm` сумма должна быть `1.0`. Для `solve` сумма нормализуется. HU leftover 50/30 → `0.625,0.375`. WTA → `1.0`. SNG 50/30/20 → `0.5,0.3,0.2`.

`--blinds` — `SB,BB`. Типично `50,100`. Эффективная глубина = stack / BB.

`--iterations` — у `equity` число MC-симуляций (по умолчанию 100000), у `solve` итерации Nash (по умолчанию 50), у `build_cache` симуляций на пару типов рук.

`--board` — 0–5 карт стола для `equity`. Пусто = префлоп, 3 = флоп, 4 = терн, 5 = ривер.

```bash
# Подставь свои значения:
cargo run --release --bin solve -- --stacks 1500,800 --payouts 0.625,0.375 --blinds 50,100 --iterations 50 --cache equity_cache.bin
cargo run --release --bin equity -- "Ah Kd" "QQ" --board As 7c 2d --iterations 50000
```

### Быстрая проверка проекта

```bash
cargo test
cargo clippy --all-targets
cargo fmt --check
```

