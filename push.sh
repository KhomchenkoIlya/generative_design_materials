#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YELLOW='\033[1;33m'
GREEN='\033[1;92m'
RED='\033[1;31m'
CYAN='\033[1;36m'
PURPLE='\033[1;35m'
RESET='\033[0m'

cd "$PROJECT_DIR" || exit 1

echo
printf "${YELLOW}============================================================${RESET}\n"
printf "${YELLOW}              СОЗДАНИЕ КОММИТА И ОТПРАВКА${RESET}\n"
printf "${YELLOW}============================================================${RESET}\n"
echo

printf "${CYAN}Скрипт выполнит следующие действия:${RESET}\n"
echo
echo "  • покажет изменённые и новые файлы;"
echo "  • запросит название коммита;"
echo "  • добавит все изменения командой git add -A;"
echo "  • создаст коммит;"
echo "  • отправит текущую ветку в удалённый репозиторий."
echo

if ! command -v git >/dev/null 2>&1; then
    printf "${RED}Ошибка: команда git не найдена.${RESET}\n"
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf "${RED}Ошибка: папка не является Git-репозиторием:${RESET}\n"
    echo "$PROJECT_DIR"
    exit 1
fi

BRANCH="$(git branch --show-current)"

if [[ -z "$BRANCH" ]]; then
    printf "${RED}Ошибка: репозиторий находится в состоянии detached HEAD.${RESET}\n"
    exit 1
fi

if [[ -z "$(git status --porcelain)" ]]; then
    printf "${GREEN}Изменений для коммита нет.${RESET}\n"
    exit 0
fi

printf "${PURPLE}Текущая ветка:${RESET} %s\n" "$BRANCH"
echo
printf "${YELLOW}Изменения, которые войдут в коммит:${RESET}\n"
echo

git status --short

echo
read -r -p "Продолжить? [y/n]: " ANSWER

case "${ANSWER,,}" in
    y|yes)
        ;;
    n|no|"")
        echo
        printf "${CYAN}Операция отменена пользователем.${RESET}\n"
        exit 0
        ;;
    *)
        echo
        printf "${RED}Неизвестный ответ. Введите y или n.${RESET}\n"
        exit 1
        ;;
esac

echo
read -r -p "Введите название коммита: " COMMIT_MESSAGE

if [[ -z "${COMMIT_MESSAGE//[[:space:]]/}" ]]; then
    echo
    printf "${RED}Название коммита не может быть пустым.${RESET}\n"
    exit 1
fi

echo
printf "${YELLOW}Будет создан коммит:${RESET}\n"
printf "  %s\n" "$COMMIT_MESSAGE"
echo
printf "${YELLOW}Ветка будет отправлена:${RESET}\n"
printf "  %s\n" "$BRANCH"
echo

read -r -p "Создать коммит и выполнить push? [y/n]: " CONFIRM

case "${CONFIRM,,}" in
    y|yes)
        ;;
    n|no|"")
        echo
        printf "${CYAN}Создание коммита отменено.${RESET}\n"
        exit 0
        ;;
    *)
        echo
        printf "${RED}Неизвестный ответ. Введите y или n.${RESET}\n"
        exit 1
        ;;
esac

echo
printf "${YELLOW}Добавляю изменения в коммит...${RESET}\n"

if ! git add -A; then
    printf "${RED}Ошибка: не удалось выполнить git add -A.${RESET}\n"
    exit 1
fi

echo
printf "${YELLOW}Создаю коммит...${RESET}\n"

if ! git commit -m "$COMMIT_MESSAGE"; then
    printf "${RED}Ошибка: коммит не был создан.${RESET}\n"
    exit 1
fi

COMMIT_HASH="$(git rev-parse --short HEAD)"

echo
printf "${YELLOW}Отправляю ветку в удалённый репозиторий...${RESET}\n"

if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' \
    >/dev/null 2>&1
then
    if ! git push; then
        printf "${RED}Ошибка: git push завершился неудачно.${RESET}\n"
        exit 1
    fi
else
    if ! git remote get-url origin >/dev/null 2>&1; then
        printf "${RED}Ошибка: удалённый репозиторий origin не настроен.${RESET}\n"
        printf "${YELLOW}Коммит создан локально, но не отправлен.${RESET}\n"
        exit 1
    fi

    if ! git push -u origin "$BRANCH"; then
        printf "${RED}Ошибка: не удалось отправить ветку origin/%s.${RESET}\n" \
            "$BRANCH"
        exit 1
    fi
fi

echo
printf "${GREEN}============================================================${RESET}\n"
printf "${GREEN}             КОММИТ СОЗДАН И УСПЕШНО ОТПРАВЛЕН${RESET}\n"
printf "${GREEN}============================================================${RESET}\n"
echo
printf "Ветка:  ${GREEN}%s${RESET}\n" "$BRANCH"
printf "Коммит: ${GREEN}%s${RESET}\n" "$COMMIT_HASH"
printf "Название: ${GREEN}%s${RESET}\n" "$COMMIT_MESSAGE"
echo
printf "${CYAN}Итоговый статус репозитория:${RESET}\n"
git status -sb
