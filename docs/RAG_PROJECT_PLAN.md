# RAG Lab — 서비스·실험·배포 계획

> 2026-10-30 변경: 임베딩은 API + 영구 SQLite 캐시로 전환했다. 아래의 BGE/로컬 ONNX 임베딩 설명은 초기 계획이며, 현재 계약은 [EMBEDDING_CACHE.md](EMBEDDING_CACHE.md)가 우선한다. 실제 API 호출 전까지만 구현·모의 검증한다.


공개 데이터로 검색과 청킹을 비교하고 답변 정확도·근거·지연·비용을 재현 가능하게 측정한다. 저장소는 `songhyeop/rag-lab` (public), Python 3.12 + LlamaIndex + FastAPI, React/TypeScript/Vite, uv lock을 사용한다. 로컬 실험과 AWS 시드니의 최대 5명 Google 초대 사용자 데모가 대상이다.

## 범위와 비용

생성 모델은 Bedrock `amazon.nova-lite-v1:0`으로 고정한다. 생성 예산 $30을 로컬과 데모가 공유한다. 운영·배포 $20, 후속 실험 예비 $50은 별도다. Free 플랜을 유지하며 권한이나 공개 엔드포인트가 거부되면 오류를 기록하고 중단한다. 계정 플랜·조직 정책·OIDC는 변경하지 않는다. 업로드, 크롤러, 학습, 반복 검색 에이전트, 결제, 자동 운영 배포, LLM judge는 제외한다.

## 데이터

- SciFact/BEIR: 전체 약 5천 초록, 공식 test 질의 300개는 검색 최종 평가 전용, train은 튜닝용. SciFact는 생성 정답 평가에 사용하지 않는다.
- HotpotQA 공식 distractor: seed 42로 train 200개(튜닝), dev 200개(최종 평가)를 선택한다. 선택 질문의 모든 context와 train context를 사용해 제목·본문 중복 제거 후 약 20,000개 공유 문서를 구성한다. 반드시 **축소 코퍼스 실험**으로 표시하며 fullwiki 점수와 직접 비교하지 않는다.
- 인덱스에는 문서 ID·제목·본문만 포함한다. 질문·정답·qrels·supporting facts는 별도 labels 파일이다. 최종 평가로 설정을 고르지 않는다.
- 원본 URL·SHA256·문서 ID·split·라이선스·샘플링 seed·정규화 버전을 manifest로 고정한다. 모델 revision과 ONNX 파일 체크섬도 고정한다.
- SciFact 초록은 원출판사의 권리가 유지되며 주석 라이선스와 구분한다. HotpotQA는 CC BY-SA 4.0 및 원문 출처를 표시한다. 원본·가중치·인덱스를 Git에 넣지 않는다.

출처: https://github.com/beir-cellar/beir/wiki/Datasets-available , https://github.com/allenai/scifact , https://hotpotqa.github.io/

## 실험 계약

| 항목 | 값 |
|---|---|
| 검색 | BM25 / Dense / Hybrid / Hybrid + reranker |
| 청킹 | 기본 256, overlap 32; train에서 128/256/384 비교 |
| 임베딩 | BAAI/bge-small-en-v1.5, ONNX Runtime, 정규화, FAISS IndexFlatIP |
| 리랭커 | cross-encoder/ms-marco-MiniLM-L6-v2, ONNX Runtime |
| 후보 | BM25 50, Dense 50, RRF k=60, rerank 20 |
| 생성 | 최종 5청크, context 최대 2,000토큰, 출력 512, temperature 0 |
| 기준선 | 검색 없는 답변 / 정답 근거 답변; 동일 모델·프롬프트 |
| 언어 | 질의·답변 영어, UI 한국어 |

청크 검색 결과를 문서로 묶어 중복 제거한 후 Recall@5/10, nDCG@10, MRR@10을 산출한다. HotpotQA는 두 근거 모두 검색한 비율을 추가한다. 생성은 공식 정규화 Answer EM/F1, 문서 인용 precision/recall 및 두 근거 인용 성공률이다. 문서 인용 점수를 문장 사실성으로 표현하지 않는다. 30건 수동 검토를 남긴다. 개선이 없어도 결과와 원인을 기록한다.

JSONL·CSV·Markdown 결과에 데이터·모델 revision·프롬프트 hash·설정 hash·git commit·실행 환경, 색인 시간·크기, 단계별 latency, token·비용을 연결한다. 웜 검색 p95 3초, 답변 p95 30초가 목표이며 콜드 스타트는 분리한다. fixture 테스트는 품질 평가 결과가 아니다.

## 서비스

Google Identity Services ID token을 백엔드에서 서명·iss·aud·exp·email_verified·초대 목록으로 검증한다. 키는 `sub`, Gmail/Google Workspace 계정 최대 5명이다. 토큰은 브라우저 메모리만 사용한다. `/health`만 공개이며 `/search`, `/answer`, `/experiments`, `/usage`는 모두 인증한다. 직접 Lambda URL 호출에도 동일하다. 응답에 request ID, 데이터·인덱스·설정 버전을 담는다.

DynamoDB transaction으로 호출 전에 최대 token 비용을 예약한다. 사용자 30회/UTC일·동시 1개, 전체 동시 2개, 전체 생성 $30 제한이다. 완료 시 실제 token으로 정산하고 사용량 불명은 예약을 유지한다. SDK 자동 재시도는 끄고, 명시적 재시도도 별도 예약 없이 실행하지 않는다. 근거 부족은 유보하고 잘못된 citation ID는 오류다. 토큰·전체 prompt는 로그에 남기지 않는다.

AWS SAM: x86_64 Lambda 컨테이너, 3GB RAM, 2GB /tmp, 모델 포함, S3 immutable index 다운로드/재사용, DynamoDB, CloudWatch 7일, CloudFront + 비공개 S3 정적 UI. NAT·상시 서버·관리형 vector DB 없음. 운영비 $10/$15/$20 알림은 차단 수단이 아니다. 단가는 실행 전 공식 시드니 요금표를 확인해 파일에 고정해야 하며 값이 없으면 생성은 차단한다.

## GitHub와 배포

main + 짧은 브랜치 + PR, 필수 상태 검사, 타인 리뷰 필수 아님. CI는 AWS 인증·Bedrock 호출 없이 Ruff·타입·pytest·검색 fixture 회귀·프런트 빌드를 검사한다. main은 같은 검증 후 Linux amd64 이미지와 실제 ONNX 모델 smoke, 프런트·checksum·commit·검사 결과를 후보로 3일 보관한다. Actions SHA 고정, permissions contents:read, pull_request_target 금지.

로컬 배포는 CI run ID 필수다. 저장소·main·성공 상태·commit·checksum·현재 AWS account·시드니·index·Google 설정을 확인한다. CI 이미지를 ECR로 올리고 재빌드하지 않는다. Lambda 후보 버전 검증 후 alias 전환, 버전별 프런트 반영, 이전 image/function/frontend/index를 release manifest에 기록한다. rollback으로 복원한다. 생성 smoke는 명시적 옵션이다. SAM 인프라 변경은 change set을 먼저 보여주고 확인 후 적용한다. AWS 장기 키나 세션을 GitHub에 복사하지 않는다.

## 구현 순서와 완료 기준

1. 이 문서, manifest 계약, README, Git 제외 규칙과 공개 저장소.
2. SciFact 검색과 CI.
3. HotpotQA 축소 코퍼스·생성 기준선·인용 평가·리포트.
4. Google 로그인 UI/API·원자적 예산 제한.
5. 검증 후보·SAM·로컬 배포·smoke·실제 rollback.

새 환경 재현, 라벨 격리, 토큰·초대 차단, 병렬 예산 제한, 잘못된 CI/checksum/account/index 배포 차단, release 추적과 실제 rollback을 모두 확인해야 완료다. 외부 설정 미제공·정책 거부·실행되지 않은 평가/배포는 완료라고 표시하지 않고 docs/IMPLEMENTATION_STATUS.md에 증거와 후속 명령을 남긴다.
