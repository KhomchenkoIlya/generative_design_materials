#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
FINAL_PDF="$BUILD_DIR/${PROJECT_NAME}.pdf"

BOLD='\033[1m'
PURPLE='\033[1;35m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
RESET='\033[0m'

TMP_DIR=""
cleanup() {
    if [[ -n "${TMP_DIR:-}" && -d "$TMP_DIR" ]]; then
        rm -rf -- "$TMP_DIR"
    fi
}
trap cleanup EXIT
trap 'printf "\n%bОшибка на строке %s.%b\n" "$RED" "$LINENO" "$RESET" >&2' ERR

cd "$PROJECT_DIR"

printf "%bHiddenPower: компактная упаковка проекта%b\n\n" "$BOLD" "$RESET"
cat <<EOF_TEXT
Скрипт создаст снимок текущего чистого Git-состояния проекта.

В архив войдут содержательные файлы проекта и PACK_AUDIT.txt.
Целиком исключаются .git/, bibliography/, build/, старые архивы,
виртуальные окружения, кэши и служебный мусор LaTeX.

Итоговый PDF из build/ в архив НЕ копируется. Его число страниц,
размер и SHA256 при наличии фиксируются только в PACK_AUDIT.txt.

Исходный проект изменяться не будет.
EOF_TEXT

if ! command -v git >/dev/null 2>&1; then
    printf "%bОшибка:%b команда git не найдена.\n" "$RED" "$RESET" >&2
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf "%bОшибка:%b %s не является Git-репозиторием.\n" "$RED" "$RESET" "$PROJECT_DIR" >&2
    exit 1
fi

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
    printf "%bОшибка:%b репозиторий находится в detached HEAD.\n" "$RED" "$RESET" >&2
    exit 1
fi

# Обычные tracked- и untracked-файлы должны полностью соответствовать коммиту.
# Игнорируемые файлы (например, старые *.tar.gz) не мешают упаковке и всё равно
# исключаются из архива ниже.
STATUS_PORCELAIN="$(git status --porcelain --untracked-files=all -- .)"
if [[ -n "$STATUS_PORCELAIN" ]]; then
    printf "%bОшибка:%b перед упаковкой рабочее дерево должно быть чистым.\n" "$RED" "$RESET" >&2
    printf "%s\n" "$STATUS_PORCELAIN" >&2
    exit 1
fi

if ! git diff --check; then
    printf "%bОшибка:%b git diff --check завершился неудачно.\n" "$RED" "$RESET" >&2
    exit 1
fi

HEAD_FULL="$(git rev-parse HEAD)"
HEAD_SHORT="$(git rev-parse --short HEAD)"

printf "\nВетка:  %s\n" "$BRANCH"
printf "Commit: %s\n" "$HEAD_SHORT"
printf "Git-состояние чистое; git diff --check: OK.\n"

printf "\nПродолжить упаковку? [y/n]: "
read -r ANSWER
case "${ANSWER,,}" in
    y|yes)
        ;;
    n|no|"")
        printf "%bУпаковка отменена пользователем.%b\n" "$YELLOW" "$RESET"
        exit 0
        ;;
    *)
        printf "%bНеизвестный ответ. Упаковка отменена.%b\n" "$RED" "$RESET" >&2
        exit 1
        ;;
esac

STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
ARCHIVE_PATH="$PROJECT_DIR/${PROJECT_NAME}_compact_${STAMP}.tar.gz"
TMP_DIR="$(mktemp -d)"
STAGE_DIR="$TMP_DIR/$PROJECT_NAME"
mkdir -p -- "$STAGE_DIR"

printf "\nКопирую содержательные файлы проекта...\n"

tar -C "$PROJECT_DIR" \
    --exclude='./.git' \
    --exclude='./bibliography' \
    --exclude='./build' \
    --exclude='./backups' \
    --exclude='./archives' \
    --exclude='./HiddenPower_archives' \
    --exclude='./PACK_AUDIT.txt' \
    --exclude='*.tar.gz' \
    --exclude='*.tgz' \
    --exclude='*.zip' \
    --exclude='*.7z' \
    --exclude='*.bak_*' \
    --exclude='*.aux' \
    --exclude='*.log' \
    --exclude='*.out' \
    --exclude='*.toc' \
    --exclude='*.lof' \
    --exclude='*.lot' \
    --exclude='*.fls' \
    --exclude='*.fdb_latexmk' \
    --exclude='*.synctex.gz' \
    --exclude='*.bbl' \
    --exclude='*.blg' \
    --exclude='*.bcf' \
    --exclude='*.run.xml' \
    --exclude='*.xdv' \
    --exclude='*.idx' \
    --exclude='*.ind' \
    --exclude='*.ilg' \
    --exclude='*.nav' \
    --exclude='*.snm' \
    --exclude='*.vrb' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --exclude='*~' \
    --exclude='.DS_Store' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='node_modules' \
    -cf - . | tar -C "$STAGE_DIR" -xf -

AUDIT_FILE="$STAGE_DIR/PACK_AUDIT.txt"
{
    echo "HiddenPower compact archive audit"
    echo "================================="
    echo "Дата создания: $(date --iso-8601=seconds)"
    echo "Исходная папка: $PROJECT_DIR"
    echo "Имя архива: $(basename "$ARCHIVE_PATH")"
    echo

    echo "GIT"
    echo "---"
    echo "Ветка: $BRANCH"
    echo "Commit: $HEAD_FULL"
    echo "Короткий commit: $HEAD_SHORT"
    echo "Статус: чисто"
    echo "git diff --check: OK"
    echo

    echo "ИСКЛЮЧЕНО ИЗ АРХИВА"
    echo "--------------------"
    echo ".git/"
    echo "bibliography/"
    echo "build/"
    echo "старые архивы"
    echo "виртуальные окружения и кэши"
    echo "служебный мусор LaTeX"
    echo

    echo "СБОРКА PDF"
    echo "----------"
    if [[ -f "$FINAL_PDF" ]]; then
        echo "PDF: $FINAL_PDF"
        echo "Размер: $(du -h "$FINAL_PDF" | awk '{print $1}')"
        echo "SHA256: $(sha256sum "$FINAL_PDF" | awk '{print $1}')"
        if command -v pdfinfo >/dev/null 2>&1; then
            pages="$(pdfinfo "$FINAL_PDF" 2>/dev/null | awk -F: '/^Pages:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
            echo "Страниц: ${pages:-не удалось определить}"
        else
            echo "Страниц: pdfinfo не установлен"
        fi
    else
        echo "Итоговый PDF не найден: $FINAL_PDF"
    fi
    echo

    echo "ОТЧЁТЫ СБОРКИ"
    echo "--------------"
    if [[ -f "$BUILD_DIR/important_warnings.txt" ]]; then
        important_count="$(wc -l < "$BUILD_DIR/important_warnings.txt" | tr -d ' ')"
        echo "Важных замечаний: $important_count"
    else
        echo "Важных замечаний: отчёт отсутствует"
    fi
    if [[ -f "$BUILD_DIR/minor_warnings.txt" ]]; then
        minor_count="$(wc -l < "$BUILD_DIR/minor_warnings.txt" | tr -d ' ')"
        echo "Неважных замечаний: $minor_count"
    else
        echo "Неважных замечаний: отчёт отсутствует"
    fi
    echo

    echo "РАЗМЕРЫ"
    echo "-------"
    echo "Исходный проект: $(du -sh "$PROJECT_DIR" 2>/dev/null | awk '{print $1}')"
    echo "Компактная копия до сжатия: $(du -sh "$STAGE_DIR" 2>/dev/null | awk '{print $1}')"
    echo

    echo "ФАЙЛЫ В КОМПАКТНОЙ КОПИИ"
    echo "-------------------------"
    find "$STAGE_DIR" -type f ! -path "$AUDIT_FILE" -printf '%P\n' | LC_ALL=C sort
    echo "PACK_AUDIT.txt"
} > "$AUDIT_FILE"

printf "Создаю tar.gz...\n"
tar -C "$TMP_DIR" -czf "$ARCHIVE_PATH" "$PROJECT_NAME"

ARCHIVE_LIST="$TMP_DIR/archive_contents.txt"
tar -tzf "$ARCHIVE_PATH" > "$ARCHIVE_LIST"

FORBIDDEN_PATTERN="^${PROJECT_NAME}/(\\.git|bibliography|build)(/|$)"
if grep -Eq "$FORBIDDEN_PATTERN" "$ARCHIVE_LIST"; then
    rm -f -- "$ARCHIVE_PATH"
    printf "%bОшибка:%b в архив попал запрещённый каталог .git/, bibliography/ или build/.\n" \
        "$RED" "$RESET" >&2
    exit 1
fi

if grep -Eq "^${PROJECT_NAME}/.*\\.(tar\\.gz|tgz|zip|7z)$" "$ARCHIVE_LIST"; then
    rm -f -- "$ARCHIVE_PATH"
    printf "%bОшибка:%b в архив попал старый архив.\n" "$RED" "$RESET" >&2
    exit 1
fi

if ! grep -Fxq "${PROJECT_NAME}/PACK_AUDIT.txt" "$ARCHIVE_LIST"; then
    rm -f -- "$ARCHIVE_PATH"
    printf "%bОшибка:%b PACK_AUDIT.txt отсутствует в готовом архиве.\n" "$RED" "$RESET" >&2
    exit 1
fi

# Повторное чтение архива проверяет целостность gzip/tar после создания.
tar -tzf "$ARCHIVE_PATH" >/dev/null

ARCHIVE_SIZE="$(du -h "$ARCHIVE_PATH" | awk '{print $1}')"
FILE_COUNT="$(grep -vc '/$' "$ARCHIVE_LIST" || true)"
SHA256="$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')"

printf "\n%b========== СКОПИРУЙ И ПРИШЛИ ЭТОТ БЛОК ==========%b\n" "$PURPLE" "$RESET"
printf "ветка:        %s\n" "$BRANCH"
printf "commit:       %s\n" "$HEAD_SHORT"
printf "архив:        %s\n" "$ARCHIVE_PATH"
printf "размер:       %s\n" "$ARCHIVE_SIZE"
printf "файлов:       %s\n" "$FILE_COUNT"
printf "sha256:       %s\n" "$SHA256"
printf "аудит внутри: %s/PACK_AUDIT.txt\n" "$PROJECT_NAME"
printf "%b===================================================%b\n" "$PURPLE" "$RESET"
printf "\n%bГотово.%b\n" "$GREEN" "$RESET"
