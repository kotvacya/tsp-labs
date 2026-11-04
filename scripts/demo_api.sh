#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

print_json() {
    if command -v jq >/dev/null 2>&1; then
        jq .
    else
        python3 -m json.tool
    fi
}

echo "=== 1. Health ==="
curl -s -X GET "$BASE_URL/health" | print_json

echo "=== 2. Register ==="
TIMESTAMP=$(date +%s)
OP_EMAIL="operator_${TIMESTAMP}@example.com"
RES_EMAIL="researcher_${TIMESTAMP}@example.com"
ADM_EMAIL="admin_${TIMESTAMP}@example.com"

curl -s -X POST "$BASE_URL/api/v1/auth/register" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$OP_EMAIL\",\"password\":\"operator_pass123\",\"full_name\":\"Test Operator\",\"role\":\"OPERATOR\"}" | print_json

curl -s -X POST "$BASE_URL/api/v1/auth/register" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$RES_EMAIL\",\"password\":\"researcher_pass123\",\"full_name\":\"Test Researcher\",\"role\":\"RESEARCHER\"}" | print_json

curl -s -X POST "$BASE_URL/api/v1/auth/register" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$ADM_EMAIL\",\"password\":\"admin_pass123\",\"full_name\":\"Test Admin\",\"role\":\"ADMIN\"}" | print_json

echo "=== 3. Login ==="
OP_TOKEN=$(curl -s -X POST "$BASE_URL/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$OP_EMAIL\",\"password\":\"operator_pass123\"}" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

RES_TOKEN=$(curl -s -X POST "$BASE_URL/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$RES_EMAIL\",\"password\":\"researcher_pass123\"}" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

ADM_TOKEN=$(curl -s -X POST "$BASE_URL/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$ADM_EMAIL\",\"password\":\"admin_pass123\"}" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

echo "=== 4. Auth Me ==="
curl -s -w "\nstatus: %{http_code}\n" -X GET "$BASE_URL/api/v1/auth/me"
curl -s -X GET "$BASE_URL/api/v1/auth/me" -H "Authorization: Bearer $RES_TOKEN" | print_json

echo "=== 5. Dataset RBAC ==="
curl -s -w "\nstatus: %{http_code}\n" -X POST "$BASE_URL/api/v1/datasets" \
    -H "Authorization: Bearer $RES_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"name":"Restricted Dataset","storage_dir":"/data/test","total_samples":100}'

DATASET_RES=$(curl -s -X POST "$BASE_URL/api/v1/datasets" \
    -H "Authorization: Bearer $OP_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"NYU-Depth-V2-$TIMESTAMP\",\"storage_dir\":\"/datasets/nyu_$TIMESTAMP\",\"total_samples\":1449}")
echo "$DATASET_RES" | print_json
DATASET_ID=$(echo "$DATASET_RES" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

echo "=== 6. Models ==="
MODEL1_RES=$(curl -s -X POST "$BASE_URL/api/v1/models" \
    -H "Authorization: Bearer $RES_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"DPT-Large-$TIMESTAMP\",\"architecture\":\"VisionTransformer\",\"backbone\":\"ViT-L/16\",\"weights_rel_path\":\"/weights/dpt_$TIMESTAMP.pt\",\"is_active\":true}")
echo "$MODEL1_RES" | print_json
MODEL1_ID=$(echo "$MODEL1_RES" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

MODEL2_RES=$(curl -s -X POST "$BASE_URL/api/v1/models" \
    -H "Authorization: Bearer $RES_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"AdaBins-Trans-$TIMESTAMP\",\"architecture\":\"Transformer\",\"backbone\":\"EfficientNet-B5\",\"weights_rel_path\":\"/weights/adabins_$TIMESTAMP.pt\",\"is_active\":true}")
echo "$MODEL2_RES" | print_json
MODEL2_ID=$(echo "$MODEL2_RES" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

echo "=== 7. Experiment ==="
EXP_RES=$(curl -s -X POST "$BASE_URL/api/v1/experiments" \
    -H "Authorization: Bearer $RES_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"Indoor Depth Evaluation $TIMESTAMP\",\"dataset_id\":$DATASET_ID,\"model_ids\":[$MODEL1_ID,$MODEL2_ID]}")
echo "$EXP_RES" | print_json
EXP_ID=$(echo "$EXP_RES" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

echo "=== 8. Leaderboard ==="
curl -s -X GET "$BASE_URL/api/v1/experiments/$EXP_ID/leaderboard?sort_by=rmse&order=asc" | print_json

echo "=== 9. Model Delete RBAC ==="
curl -s -w "\nstatus: %{http_code}\n" -X DELETE "$BASE_URL/api/v1/models/$MODEL2_ID" -H "Authorization: Bearer $RES_TOKEN"
curl -s -w "\nstatus: %{http_code}\n" -X DELETE "$BASE_URL/api/v1/models/$MODEL2_ID" -H "Authorization: Bearer $ADM_TOKEN"
curl -s -w "\nstatus: %{http_code}\n" -X GET "$BASE_URL/api/v1/models/$MODEL2_ID"
