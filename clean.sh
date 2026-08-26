#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
FINAL_PDF="$BUILD_DIR/${PROJECT_NAME}.pdf"

YELLOW='\033[1;33m'
GREEN='\033[1;92m'
RED='\033[1;31m'
CYAN='\033[1;36m'
RESET='\033[0m'

cd "$PROJECT_DIR"

if ! command -v git >/dev/null 2>&1; then
    printf "${RED}Ошибка: команда git не найдена. Без Git безопасная очистка запрещена.${RESET}\n" >&2
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf "${RED}Ошибка: папка проекта не является Git-репозиторием.${RESET}\n" >&2
    printf "${RED}Без этого нельзя гарантировать сохранность отслеживаемых файлов.${RESET}\n" >&2
    exit 1
fi

TEMP_FILES=()
PROTECTED_TRACKED=()
declare -A SEEN=()

is_tracked() {
    local file="$1"
    local relative
    relative="${file#"$PROJECT_DIR"/}"
    git ls-files --error-unmatch -- "$relative" >/dev/null 2>&1
}

add_candidate() {
    local file="$1"

    [[ -f "$file" ]] || return 0

    if [[ -n "${SEEN[$file]:-}" ]]; then
        return 0
    fi
    SEEN["$file"]=1

    if is_tracked "$file"; then
        PROTECTED_TRACKED+=("$file")
    else
        TEMP_FILES+=("$file")
    fi
}

for extension in \
    aux log out toc lof lot \
    bbl blg bcf run.xml \
    fls fdb_latexmk synctex.gz \
    xdv idx ind ilg nav snm vrb
do
    add_candidate "$PROJECT_DIR/main.$extension"
done

# Итоговый PDF живёт только в build/. Случайный main.pdf в корне — мусор,
# если пользователь явно не добавил его в Git.
add_candidate "$PROJECT_DIR/main.pdf"

if [[ -d "$BUILD_DIR" ]]; then
    while IFS= read -r -d '' file; do
        add_candidate "$file"
    done < <(
        find "$BUILD_DIR" -type f \
            ! -path "$FINAL_PDF" \
            \( \
                -name '*.aux' \
                -o -name '*.log' \
                -o -name '*.out' \
                -o -name '*.toc' \
                -o -name '*.lof' \
                -o -name '*.lot' \
                -o -name '*.bbl' \
                -o -name '*.blg' \
                -o -name '*.bcf' \
                -o -name '*.run.xml' \
                -o -name '*.fls' \
                -o -name '*.fdb_latexmk' \
                -o -name '*.synctex.gz' \
                -o -name '*.xdv' \
                -o -name '*.idx' \
                -o -name '*.ind' \
                -o -name '*.ilg' \
                -o -name '*.nav' \
                -o -name '*.snm' \
                -o -name '*.vrb' \
                -o -name 'main.pdf' \
                -o -name 'pass*.console.log' \
            \) \
            -print0
    )
fi

echo
printf "${YELLOW}============================================================${RESET}\n"
printf "${YELLOW}               ОЧИСТКА ПРОЕКТА HIDDENPOWER${RESET}\n"
printf "${YELLOW}============================================================${RESET}\n"
echo

printf "${CYAN}Скрипт удаляет только неотслеживаемый служебный мусор LaTeX.${RESET}\n"
echo "Ни один файл, известный Git, удалён не будет."
echo

printf "${GREEN}Всегда сохраняются:${RESET}\n"
echo "  • все Git-tracked файлы проекта;"
echo "  • main.tex, preamble.tex, sections/, figures/, code/, bibliography/;"
echo "  • build/important_warnings.txt и build/minor_warnings.txt;"
echo "  • итоговый PDF: $FINAL_PDF"
echo

if (( ${#PROTECTED_TRACKED[@]} > 0 )); then
    printf "${CYAN}Найдены служебно выглядящие, но отслеживаемые Git файлы; они защищены:${RESET}\n"
    for file in "${PROTECTED_TRACKED[@]}"; do
        printf '  • %s\n' "${file#"$PROJECT_DIR"/}"
    done
    echo
fi

if (( ${#TEMP_FILES[@]} == 0 )); then
    printf "${GREEN}Неотслеживаемый служебный мусор не найден. Очищать нечего.${RESET}\n"
    exit 0
fi

printf "${YELLOW}Будут удалены %d файлов:${RESET}\n" "${#TEMP_FILES[@]}"
for file in "${TEMP_FILES[@]}"; do
    printf '  • %s\n' "${file#"$PROJECT_DIR"/}"
done

echo
read -r -p "Продолжить очистку? [y/n]: " ANSWER

case "${ANSWER,,}" in
    y|yes)
        ;;
    n|no|"")
        printf "${CYAN}Очистка отменена пользователем.${RESET}\n"
        exit 0
        ;;
    *)
        printf "${RED}Неизвестный ответ. Очистка отменена.${RESET}\n" >&2
        exit 1
        ;;
esac

DELETED_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

for file in "${TEMP_FILES[@]}"; do
    [[ -f "$file" ]] || continue

    # Повторная проверка непосредственно перед rm защищает от изменения
    # состояния Git между preview и подтверждением пользователя.
    if is_tracked "$file"; then
        ((SKIPPED_COUNT += 1))
        printf "${CYAN}Пропущен как Git-tracked: %s${RESET}\n" "${file#"$PROJECT_DIR"/}"
        continue
    fi

    if rm -f -- "$file"; then
        ((DELETED_COUNT += 1))
    else
        ((FAILED_COUNT += 1))
        printf "${RED}Не удалось удалить: %s${RESET}\n" "${file#"$PROJECT_DIR"/}" >&2
    fi
done

if [[ -d "$BUILD_DIR" ]]; then
    find "$BUILD_DIR" -mindepth 1 -type d -empty -delete
fi

echo
printf "${GREEN}============================================================${RESET}\n"
printf "${GREEN}                    ОЧИСТКА ЗАВЕРШЕНА${RESET}\n"
printf "${GREEN}============================================================${RESET}\n"
printf "Удалено файлов: ${GREEN}%d${RESET}\n" "$DELETED_COUNT"
printf "Защищено Git при удалении: ${GREEN}%d${RESET}\n" "$SKIPPED_COUNT"

if (( FAILED_COUNT == 0 )); then
    printf "Ошибок удаления: ${GREEN}0${RESET}\n"
else
    printf "Ошибок удаления: ${RED}%d${RESET}\n" "$FAILED_COUNT"
fi

if [[ -f "$FINAL_PDF" ]]; then
    printf "${GREEN}Итоговый PDF сохранён:${RESET} %s\n" "$FINAL_PDF"
else
    printf "${YELLOW}Итоговый PDF пока отсутствует.${RESET}\n"
fi

if (( FAILED_COUNT > 0 )); then
    exit 1
fi
