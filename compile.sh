#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

MAIN_FILE="$PROJECT_DIR/main.tex"
BUILD_DIR="$PROJECT_DIR/build"
LOG_FILE="$BUILD_DIR/main.log"

FINAL_PDF="$BUILD_DIR/${PROJECT_NAME}.pdf"
IMPORTANT_REPORT="$BUILD_DIR/important_warnings.txt"
MINOR_REPORT="$BUILD_DIR/minor_warnings.txt"

cd "$PROJECT_DIR" || exit 1

echo
echo "Будут выполнены следующие действия:"
echo
echo "  1. Создана папка build и необходимые вложенные каталоги."
echo "  2. main.tex будет собран XeLaTeX от двух до четырёх раз — до стабилизации ссылок."
echo "  3. Итоговый PDF будет назван ${PROJECT_NAME}.pdf."
echo "  4. PDF и журналы сборки будут сохранены в build."
echo "  5. Журнал будет проверен на ошибки и предупреждения."
echo
echo "Важными считаются:"
echo "  ошибки компиляции, пропавшие символы, неопределённые ссылки,"
echo "  проблемы библиографии, повторяющиеся метки и переполненные блоки."
echo
echo "Неважными считаются:"
echo "  underfull-блоки и прочие некритические предупреждения оформления."
echo

read -r -p "Продолжить сборку? [y/n]: " ANSWER

case "${ANSWER,,}" in
    y|yes)
        ;;
    n|no|"")
        echo "Сборка отменена."
        exit 0
        ;;
    *)
        echo "Неизвестный ответ. Сборка отменена."
        exit 1
        ;;
esac

echo

if ! command -v xelatex >/dev/null 2>&1; then
    echo "ОШИБКА: команда xelatex не найдена."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ОШИБКА: команда python3 не найдена."
    exit 1
fi

if [[ ! -f "$MAIN_FILE" ]]; then
    echo "ОШИБКА: файл main.tex не найден:"
    echo "$MAIN_FILE"
    exit 1
fi

mkdir -p "$BUILD_DIR"

# XeLaTeX должен иметь возможность создавать вспомогательные файлы
# для \include{sections/...} внутри build.
while IFS= read -r -d '' SOURCE_DIR; do
    if [[ "$SOURCE_DIR" == "$PROJECT_DIR" ]]; then
        continue
    fi

    RELATIVE_DIR="${SOURCE_DIR#"$PROJECT_DIR"/}"
    mkdir -p "$BUILD_DIR/$RELATIVE_DIR"
done < <(
    find "$PROJECT_DIR" \
        \( -path "$BUILD_DIR" -o -path "$PROJECT_DIR/.git" \) -prune \
        -o -type d -print0
)

rm -f \
    "$BUILD_DIR/main.pdf" \
    "$FINAL_PDF" \
    "$IMPORTANT_REPORT" \
    "$MINOR_REPORT" \
    "$BUILD_DIR/pass1.console.log" \
    "$BUILD_DIR/pass2.console.log"

run_xelatex_pass() {
    local PASS_NUMBER="$1"
    local CONSOLE_LOG="$BUILD_DIR/pass${PASS_NUMBER}.console.log"

    echo "XeLaTeX: проход ${PASS_NUMBER} из ${MAX_XELATEX_PASSES:-2}..."

    xelatex \
        -interaction=nonstopmode \
        -halt-on-error \
        -file-line-error \
        -output-directory="$BUILD_DIR" \
        "$MAIN_FILE" \
        >"$CONSOLE_LOG" 2>&1
}

BUILD_STATUS=0
MAX_XELATEX_PASSES=4
PASS_COUNT=0
RERUN_PATTERN='Rerun to get cross-references right|Label\(s\) may have changed|There were undefined references'

for PASS_NUMBER in $(seq 1 "$MAX_XELATEX_PASSES"); do
    if ! run_xelatex_pass "$PASS_NUMBER"; then
        BUILD_STATUS=1
        break
    fi

    PASS_COUNT="$PASS_NUMBER"

    if [[ "$PASS_NUMBER" -ge 2 ]] && \
       ! grep -Eq "$RERUN_PATTERN" "$LOG_FILE"; then
        break
    fi
done

if [[ "$BUILD_STATUS" -eq 0 ]]; then
    echo "XeLaTeX: ссылки стабилизированы за ${PASS_COUNT} проход(а)."
fi

# Разбор итогового журнала.
python3 - "$LOG_FILE" "$IMPORTANT_REPORT" "$MINOR_REPORT" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
important_path = Path(sys.argv[2])
minor_path = Path(sys.argv[3])

if not log_path.exists():
    important_path.write_text(
        "[1] Журнал main.log не был создан.\n",
        encoding="utf-8",
    )
    minor_path.write_text("", encoding="utf-8")
    raise SystemExit(0)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

important_patterns = [
    r"LaTeX Error:",
    r"Package .* Error:",
    r"Class .* Error:",
    r"Undefined control sequence",
    r"Emergency stop",
    r"Fatal error",
    r"File .* not found",
    r"Missing character:",
    r"Citation .* undefined",
    r"Reference .* undefined",
    r"There were undefined references",
    r"There were undefined citations",
    r"multiply defined",
    r"destination with the same identifier",
    r"Overfull \\[hv]box",
    r"Please .*run Biber",
    r"Please .*run BibTeX",
    r"Rerun to get cross-references right",
    r"Label\(s\) may have changed",
]

minor_patterns = [
    r"Underfull \\[hv]box",
    r"LaTeX Warning:",
    r"Package .* Warning:",
    r"Class .* Warning:",
    r"Font Warning:",
    r"pdfTeX warning",
]

important_regex = re.compile(
    "|".join(f"(?:{pattern})" for pattern in important_patterns),
    re.IGNORECASE,
)

minor_regex = re.compile(
    "|".join(f"(?:{pattern})" for pattern in minor_patterns),
    re.IGNORECASE,
)

file_line_regex = re.compile(
    r"(?P<file>(?:\./)?[^:\s]+\.tex):(?P<line>\d+):"
)
input_line_regex = re.compile(r"on input line (?P<line>\d+)", re.I)
lines_regex = re.compile(
    r"at lines? (?P<start>\d+)(?:--(?P<end>\d+))?",
    re.I,
)
tex_file_regex = re.compile(r"\((?:\./)?([^()\s]+\.tex)")

important: list[str] = []
minor: list[str] = []

seen_important: set[str] = set()
seen_minor: set[str] = set()

current_tex = "main.tex"


def clean_message(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def locate(message: str, log_line: int) -> str:
    match = file_line_regex.search(message)
    if match:
        return f"{match.group('file')}:{match.group('line')}"

    match = input_line_regex.search(message)
    if match:
        return f"{current_tex}:{match.group('line')}"

    match = lines_regex.search(message)
    if match:
        start = match.group("start")
        end = match.group("end")
        source_lines = start if end is None else f"{start}--{end}"
        return f"{current_tex}: строки {source_lines}"

    return f"main.log:{log_line}"


for index, raw_line in enumerate(lines, start=1):
    for match in tex_file_regex.finditer(raw_line):
        current_tex = match.group(1)

    message = clean_message(raw_line)
    if not message:
        continue

    location = locate(message, index)
    entry = f"{location} — {message}"

    if important_regex.search(message):
        if entry not in seen_important:
            seen_important.add(entry)
            important.append(entry)
        continue

    if minor_regex.search(message):
        if entry not in seen_minor:
            seen_minor.add(entry)
            minor.append(entry)


def write_report(path: Path, entries: list[str]) -> None:
    if not entries:
        path.write_text("", encoding="utf-8")
        return

    text = "\n".join(
        f"[{number}] {entry}"
        for number, entry in enumerate(entries, start=1)
    )
    path.write_text(text + "\n", encoding="utf-8")


write_report(important_path, important)
write_report(minor_path, minor)
PY

IMPORTANT_COUNT="$(wc -l < "$IMPORTANT_REPORT" | tr -d ' ')"
MINOR_COUNT="$(wc -l < "$MINOR_REPORT" | tr -d ' ')"

echo
echo "================================================================"
echo "ИТОГ СБОРКИ"
echo "================================================================"

if [[ "$BUILD_STATUS" -ne 0 || ! -f "$BUILD_DIR/main.pdf" ]]; then
    echo
    echo "PDF не создан: XeLaTeX завершился с ошибкой."
    echo
    echo "Важных замечаний: $IMPORTANT_COUNT"

    if [[ "$IMPORTANT_COUNT" -gt 0 ]]; then
        cat "$IMPORTANT_REPORT"
    fi

    echo
    echo "Неважных замечаний: $MINOR_COUNT"

    if [[ "$MINOR_COUNT" -gt 0 ]]; then
        cat "$MINOR_REPORT"
    fi

    echo
    echo "Полный журнал:"
    echo "$LOG_FILE"
    echo
    echo "Вывод первого прохода:"
    echo "$BUILD_DIR/pass1.console.log"

    if [[ -f "$BUILD_DIR/pass2.console.log" ]]; then
        echo
        echo "Вывод второго прохода:"
        echo "$BUILD_DIR/pass2.console.log"
    fi

    exit 1
fi

mv -f "$BUILD_DIR/main.pdf" "$FINAL_PDF"

PAGE_COUNT="не удалось определить"

if command -v pdfinfo >/dev/null 2>&1; then
    PAGE_COUNT="$(
        pdfinfo "$FINAL_PDF" 2>/dev/null |
        awk -F ':' '/^Pages:/ {
            gsub(/[[:space:]]/, "", $2)
            print $2
        }'
    )"
elif command -v qpdf >/dev/null 2>&1; then
    PAGE_COUNT="$(qpdf --show-npages "$FINAL_PDF" 2>/dev/null)"
elif command -v mutool >/dev/null 2>&1; then
    PAGE_COUNT="$(
        mutool info "$FINAL_PDF" 2>/dev/null |
        awk '/^Pages:/ {print $2; exit}'
    )"
fi

[[ -n "$PAGE_COUNT" ]] || PAGE_COUNT="не удалось определить"

echo
echo "PDF успешно создан:"
echo "$FINAL_PDF"
echo
echo "Количество страниц: $PAGE_COUNT"
echo
echo "Важных замечаний: $IMPORTANT_COUNT"

if [[ "$IMPORTANT_COUNT" -gt 0 ]]; then
    cat "$IMPORTANT_REPORT"
else
    echo "Нет."
fi

echo
echo "Неважных замечаний: $MINOR_COUNT"

if [[ "$MINOR_COUNT" -gt 0 ]]; then
    cat "$MINOR_REPORT"
else
    echo "Нет."
fi

echo
echo "Подробные отчёты:"
echo "$IMPORTANT_REPORT"
echo "$MINOR_REPORT"
echo
echo "Полный журнал XeLaTeX:"
echo "$LOG_FILE"
echo
