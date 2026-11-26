# Amazon Top-K Recommendation

## 인기 상품 추천 검증

대회에서 받은 `train.csv`를 `data/train.csv`에 두고 저장소 루트에서 실행합니다. Python 표준 라이브러리만 사용합니다.

```bash
python3 popularity.py
```

다른 경로를 쓸 때는 `python3 popularity.py --train /path/to/train.csv --output outputs`로 실행합니다.

- 사용자-상품 쌍을 중복 제거하고, 시드 42로 사용자마다 약 10%를 검증용으로 분리합니다. 학습 기록은 사용자마다 최소 1개 남깁니다.
- 남은 기록에서 상품별 사용자 수를 세고, 각 사용자가 학습 분할에서 이미 본 상품을 제외해 상위 20개를 추천합니다. 동률은 상품 ID 오름차순으로 처리합니다.
- 공식 Recall@20 집계 방식으로 1,371 ÷ 59,036 = **0.023223**을 재현했습니다. 학습 기록은 556,847건, 숨긴 기록은 59,095건입니다.
- `outputs/`에 `validation_train.csv`, `validation_truth.csv`, `validation_recommendations.csv`가 생성됩니다. 추천 파일은 로컬 검증용입니다.
