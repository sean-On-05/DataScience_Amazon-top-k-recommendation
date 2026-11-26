# Amazon Top-K Recommendation

## 공식 대회 베이스라인

Kaggle 리더보드의 `LightGCN (Baseline)` 점수는 **Recall@20 0.04187 (4.187%)**입니다. 과제 공지에서 베이스라인 통과 배점은 40점이며, 동점 인정 여부는 확인되지 않았습니다. 통과 여부는 Kaggle 제출 점수로 확인합니다. [Kaggle 리더보드](https://www.kaggle.com/competitions/ssu-data-science-amazon-top-k-recommendation/leaderboard)

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

## 아이템 기반 협업 필터링

`data/`에 대회 `train.csv`, `test.csv`, `sample_submission.csv`를 둔 뒤 실행합니다.

```bash
python3 -m pip install numpy scipy
python3 popularity.py
python3 item_cf.py
```

두 상품을 모두 리뷰한 사용자 수를 각 상품의 리뷰 사용자 수로 코사인 정규화합니다. 사용자가 이미 리뷰한 상품 각각과의 유사도를 합산해 상위 20개를 추천합니다. 후보가 부족하면 인기 상품으로 채웁니다. 검증은 시드 42의 기존 분할을 사용하고, Kaggle 제출용 CSV는 전체 학습 기록으로 다시 계산해 `outputs/item_cf_submission.csv`에 저장합니다.

| 모델 | 공통 검증 Recall@20 | Kaggle Recall@20 |
|---|---:|---:|
| 인기 상품 추천 | 0.023223 | 0.02292 |
| 아이템 기반 협업 필터링 | 0.033082 | 0.04076 |
| TF-IDF 상품명 | 0.023460 | — |
| TF-IDF 상품명 + 카테고리 | 0.021309 | — |
| 공식 LightGCN | — | 0.04187 |

Kaggle에서 아이템 기반 협업 필터링은 인기 상품 추천보다 0.01784 높았고, 공식 베이스라인까지 0.00111 남았습니다.

협업 필터링은 공통 검증셋에서 59,036개 중 1,953개를 맞혔습니다. [Kaggle 제출용 CSV](item_cf_submission.csv)

## 콘텐츠 추천

`data/item_meta.jsonl`을 두고 아래 명령을 실행합니다. 상품명 단독과 상품명·카테고리 결합의 Recall@20 및 메타데이터 결측 통계가 출력됩니다.

```bash
python3 -m pip install numpy scipy scikit-learn
python3 popularity.py
python3 content_model.py
```

[콘텐츠 추천 보고서 초안](content_report.md)

## 아이템 협업 필터링 개선 실험

동일한 검증 분할에서 상품별 이웃 수 제한과 공동 리뷰 수에 따른 유사도 감쇠를 비교했습니다. 감쇠는 기존 코사인 유사도에 `n / (n + 5)`를 곱하는 방식이며, `n`은 두 상품을 모두 리뷰한 사용자 수입니다. 이웃 제한을 적용할 때는 조정된 유사도 순으로 상품별 상위 이웃만 남깁니다.

| 상품별 이웃 수 | 감쇠 상수 | 공통 검증 적중 수 | 공통 검증 Recall@20 |
|---|---:|---:|---:|
| 전체 | 0 | 1,953 | 0.033082 |
| 100 | 0 | 947 | 0.016041 |
| 300 | 0 | 1,146 | 0.019412 |
| 전체 | 5 | 2,726 | **0.046175** |
| 100 | 5 | 2,494 | 0.042245 |
| 300 | 5 | 2,611 | 0.044227 |

감쇠만 적용했을 때 기존 모델보다 773개를 더 맞혔습니다. 테스트한 이웃 수 제한은 같은 감쇠 설정에서 성능을 낮췄습니다.

```bash
python3 popularity.py
python3 item_cf.py --shrinkage 5 --output outputs/item_cf_shrink5_submission.csv
```

[감쇠 모델 Kaggle 제출용 CSV](item_cf_shrink5_submission.csv)

## EASE 실험

[EASE 논문](https://arxiv.org/abs/1905.03375)의 선형 모델을 이진 사용자–상품 행렬에 적용했습니다. 정규화한 상품–상품 행렬의 역행렬을 구해 가중치를 계산하고, 사용자가 이미 본 상품은 추천에서 제외했습니다. 모든 설정은 시드 42의 공통 검증 분할에서 비교했습니다.

| 모델 | 정규화 λ | 검증 적중 수 | 공통 검증 Recall@20 |
|---|---:|---:|---:|
| 아이템 CF 감쇠 | — | 2,726 | 0.046175 |
| EASE | 50 | 2,469 | 0.041822 |
| EASE | 200 | 2,766 | 0.046853 |
| EASE | 500 | 2,839 | **0.048089** |

검증에서 최고였던 λ=500으로 전체 학습 데이터에 다시 적합해 제출용 CSV를 생성했습니다. 이 실험은 NumPy와 SciPy를 사용하며, 상품 수에 비례해 제곱 크기의 밀집 행렬을 할당합니다.

```bash
python3 popularity.py
OPENBLAS_NUM_THREADS=2 python3 ease.py
```

[EASE Kaggle 제출용 CSV](ease_submission.csv)
