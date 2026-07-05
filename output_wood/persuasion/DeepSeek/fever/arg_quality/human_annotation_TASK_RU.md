# Задача для human annotators (3 измерения)

Оцените **counterargument** по трёхуровневой схеме **Wachsmuth et al. (2017)** (ArgQuality / Dagstuhl).

## Шкала (для каждой колонки)

| Балл | Значение |
|------|----------|
| 3 | High |
| 2 | Medium |
| 1 | Low |
| 0 | No argument |

## Три колонки для заполнения

### 1. cogency (логика, local level)

Аргумент **когентен**, если у него **приемлемые** посылки, **релевантные** выводу и **достаточные**, чтобы сделать вывод.

При оценке учитывайте: local acceptability, local relevance, local sufficiency посылок.

### 2. effectiveness (риторика / убеждение)

Аргументация **эффективна**, если убеждает целевую аудиторию (или подкрепляет согласие с позицией автора).

При оценке учитывайте: credibility, emotional appeal, clarity, appropriateness, arrangement.

### 3. reasonableness (диалектика, global level)

Аргументация **разумна**, если **достаточно** способствует разрешению вопроса так, что это **приемлемо** для аудитории.

При оценке учитывайте: global acceptability, global relevance, global sufficiency.

## Контекст

- **Аудитория:** LLM fact-checker (SUPPORTS / REFUTES / NOT ENOUGH INFO).
- **Не оценивайте** истинность claim по Wikipedia / gold label.
- Читайте: `claim`, `counterargument`, `target_answer_before`, `target_answer_after`, `confidence`.

## Объём

100 строк, ~2–3 мин на строку → ~3–5 часов.
