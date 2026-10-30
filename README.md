# RAG Lab

공개 데이터에서 검색·청킹·근거 인용·지연·비용을 재현 가능하게 비교하는 실험실입니다. 영어 질의/답변과 한국어 UI를 사용합니다.

- BM25 / API embeddings + 영구 벡터 캐시 + FAISS / RRF Hybrid / ONNX reranker
- SciFact 5,183문서와 HotpotQA 축소 코퍼스 20,000문서
- 문서 단위 Recall·nDCG·MRR, Answer EM/F1·문서 인용 평가
- Google 초대 로그인, DynamoDB 원자 예산, Bedrock Nova Lite
- AWS 인증 없는 PR CI, immutable CI 후보, 로컬 SAM 검토·배포·rollback

```sh
uv sync --frozen --extra data
uv run pytest -q  # 모의 API 응답, 실제 임베딩 호출 없음
uv run rag cache-info
```

임베딩은 API 제공자를 명시적으로 고른 뒤 사용합니다. 기본은 호출 차단이며, 저장한 벡터로 색인을 다시 만들 때도 API를 호출하지 않습니다. [캐시 재사용·API 실행 계약](docs/EMBEDDING_CACHE.md)을 먼저 읽으세요.

Python 3.12, uv, Node 22 이상이 필요합니다. 데이터·모델·계정 설정은 저장소에 포함하지 않습니다. 실제 생성에는 Google 인증·공유 DynamoDB 예산·검증한 시드니 단가가 필요합니다.

[계획](docs/RAG_PROJECT_PLAN.md) · [데이터 계약](docs/DATA_CONTRACT.md) · [실행·배포 방법](docs/RUNBOOK.md) · [실제 검증 상태와 남은 작업](docs/IMPLEMENTATION_STATUS.md) · [실험 리포트](docs/reports)

현재 Google OAuth 설정과 인프라 적용 확인을 기다리고 있습니다. 실제 생성 평가와 AWS 데모·rollback은 아직 검증하지 않았습니다. HotpotQA 결과는 축소 코퍼스 실험이며 공식 fullwiki 결과와 직접 비교할 수 없습니다. [데이터·모델 출처 및 라이선스](THIRD_PARTY_NOTICES.md)를 확인하세요. MIT는 자체 작성 코드에 적용되며 원본 데이터의 이용 조건을 대체하지 않습니다.
