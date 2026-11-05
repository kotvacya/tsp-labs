#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost}"
API_URL="$BASE_URL/api/v1"
MAX_SAMPLES="${MAX_SAMPLES:-3}"

print_json() {
    if command -v jq >/dev/null 2>&1; then
        jq .
    else
        python3 -m json.tool
    fi
}

echo "======================================================================"
echo "ДЕМОНСТРАЦИЯ ЛАБОРАТОРНОЙ РАБОТЫ №3: gRPC СЕРВИС ВЫЧИСЛЕНИЙ"
echo "======================================================================"

echo -e "\n--- 1. Проверка доступности API Gateway и Nginx ---"
curl -s -f "$BASE_URL/health" | print_json

echo -e "\n--- 2. Проверка gRPC Compute Engine через API Gateway ---"
curl -s -f "$API_URL/experiments/compute/status" | print_json

echo -e "\n--- 3. Проверка раздачи файлов датасета через Nginx (/datasets/) ---"
curl -s -I "$BASE_URL/datasets/nyu_depth_v2/extracted/rgb/sample_0000.png" | head -n 5

echo -e "\n--- 4. Аутентификация исследователя (JWT) ---"
TOKEN=$(curl -s -X POST "$API_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"email":"researcher@mdebench.com","password":"researcherpass"}' | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
echo "Получен JWT токен: ${TOKEN:0:30}..."

echo -e "\n--- 5. Получение предзаполненных датасетов и моделей ---"
DATASETS_JSON=$(curl -s -X GET "$API_URL/datasets")
echo "Доступные датасеты:"
echo "$DATASETS_JSON" | print_json

NYU_ID=$(echo "$DATASETS_JSON" | python3 -c "import sys, json; print(next(d['id'] for d in json.load(sys.stdin) if d['name'] == 'NYU-Depth-V2'))")

MODELS_JSON=$(curl -s -X GET "$API_URL/models")
echo -e "\nДоступные модели нейросетей:"
echo "$MODELS_JSON" | print_json

M1_ID=$(echo "$MODELS_JSON" | python3 -c "import sys, json; print(next(m['id'] for m in json.load(sys.stdin) if 'Tiny-NYU' in m['name']))")
M2_ID=$(echo "$MODELS_JSON" | python3 -c "import sys, json; print(next(m['id'] for m in json.load(sys.stdin) if 'Large-NYU' in m['name']))")

echo -e "\n--- 6. Создание эксперимента для сравнения IEBins-Tiny и IEBins-Large ---"
EXP_PAYLOAD=$(printf '{"title":"IEBins Tiny vs Large Benchmark","dataset_id":%d,"model_ids":[%d,%d]}' "$NYU_ID" "$M1_ID" "$M2_ID")
EXP_RES=$(curl -s -X POST "$API_URL/experiments" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$EXP_PAYLOAD")
echo "$EXP_RES" | print_json
EXP_ID=$(echo "$EXP_RES" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

echo -e "\n--- 7. Запуск вычислений бенчмарка через gRPC (POST /experiments/$EXP_ID/run) ---"
echo "FastAPI отправляет RPC-запрос в Compute Service (samples: $MAX_SAMPLES)..."
RUN_RES=$(curl -s -X POST "$API_URL/experiments/$EXP_ID/run?max_samples=$MAX_SAMPLES" \
    -H "Authorization: Bearer $TOKEN")
echo "$RUN_RES" | print_json

echo -e "\n--- 8. Итоговая таблица лидеров (Leaderboard, ранжирование по RMSE) ---"
curl -s -X GET "$API_URL/experiments/$EXP_ID/leaderboard?sort_by=rmse&order=asc" | print_json

echo -e "\n--- 9. Демонстрация одиночного инференса произвольного изображения (POST /models/$M1_ID/predict) ---"
echo "Отправка кадра RGB в модель IEBins через gRPC с генерацией визуализации карты глубины..."
curl -s -D /tmp/pred_headers.txt -X POST "$API_URL/models/$M1_ID/predict?colorize=true&colormap=turbo&as_image=true" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@data/nyu_depth_v2/extracted/rgb/sample_0000.png;type=image/png" \
    -o demo_predicted_depth.png

grep -iE "(HTTP/|content-type|x-depth-|x-inference-|x-image-)" /tmp/pred_headers.txt || true
if [ -f demo_predicted_depth.png ]; then
    echo "Готово! Сгенерированная карта глубины сохранена в: demo_predicted_depth.png ($(wc -c < demo_predicted_depth.png) bytes)"
fi

echo -e "\n======================================================================"
echo "Демонстрация Lab 3 успешно завершена!"
echo "======================================================================"
