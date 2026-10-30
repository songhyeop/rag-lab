# API 임베딩과 영구 벡터 캐시

2026-10-30 사용자 변경: 로컬 CPU 임베딩을 중단하고 API 임베딩으로 바꾼다. 임베딩을 버리지 않고 다음 색인·실험에서 재사용한다. 현재 작업 범위는 **실제 API 호출 전의 코드·모의 검증까지**다.

## 재사용 단위

`artifacts/embedding-cache.sqlite3`에 원본 텍스트의 hash와 float32 벡터를 저장한다. 키는 provider·model·revision/cache epoch·region·dimensions·정규화·전처리 버전·청크 내용 hash의 조합이다. 문서 ID·청크 순번·색인 디렉터리·청킹 크기는 키에 넣지 않는다. 다른 데이터셋이나 청킹 설정에서도 최종 입력 텍스트가 같으면 같은 벡터를 쓴다.

같은 데이터로 재실행하면 API 호출 없이 벡터를 읽는다. 문서를 추가·수정하면 새로운 내용만 호출한다. 모델·차원·전처리·epoch를 바꾸면 별도 namespace를 사용하며 이전 벡터는 보존한다. 서로 다른 모델의 벡터를 같은 FAISS 색인에 섞지 않는다.

API batch마다 SQLite transaction을 commit한다. 종료 후 다시 실행하면 완료 batch는 재사용한다. 파일 lock으로 한 캐시를 사용하는 여러 프로세스의 중복 호출을 막는다. 이미 응답한 API와 SQLite commit 사이에서 프로세스가 종료된 경우까지 exactly-once를 보장하지는 않는다. 해당 미완료 batch는 재호출될 수 있으며 사용량 불명 예약액은 공유 예산에서 유지한다.

벡터 checksum·길이·유한값을 검증한다. 손상된 벡터를 발견하면 오류로 중단하며 몰래 재호출·재과금하지 않는다. 캐시에 원문·질문·정답·키는 저장하지 않는다. 실제 데모 질문 벡터는 메모리에서만 재사용하고 디스크에 저장하지 않는다. 명시적 로컬 공개 데이터 평가는 query 벡터도 캐시해 반복 실험의 호출을 줄인다. query 벡터가 문서 FAISS에 추가되지는 않는다.

## API 호출 기본 차단

기본 config의 provider는 `unconfigured`다. 자동 로컬 모델 fallback은 없다. 두 어댑터만 준비했다:

- `configs/embedding-openai.json`: text-embedding-3-small, 1536차원, batch 호출. `RAG_OPENAI_API_KEY` 필요.
- `configs/embedding-bedrock.json`: 시드니 Titan Text Embeddings v2, 512차원, 청크별 호출. 기존 AWS 인증 사용.

제공자는 아직 선택하지 않았다. 키와 실제 가격도 주입하지 않았다. 원본 계획의 BGE 384차원 벡터와 호환되지 않으므로 옛 인덱스는 재사용하지 않는다. API 사업자의 내부 가중치는 local revision처럼 고정할 수 없다. 선언된 cache epoch와 실제 저장 벡터 checksum을 고정하고, 원격 모델 변경을 수용할 때 명시적으로 epoch를 올린다.

`rag index`는 기본적으로 cache-only다. cache miss가 있으면 API 호출 전에 실패한다. 새 벡터가 필요한 실행에는 `--allow-api`가 명시적으로 필요하다. 웹은 `RAG_EMBEDDING_API_ENABLED=false`가 기본이다. 설정을 켜도 검증한 embedding price와 공유 DynamoDB 예산이 없으면 호출하지 않는다.

임베딩과 Nova 생성은 같은 $30 원자 예산을 공유한다. 임베딩 호출은 생성 30회/일 카운터를 소비하지 않는다. API 전 최대 8192 입력 토큰/텍스트 비용을 예약하고 실제 input token으로 정산한다. 사용량 불명은 최대 예약 유지, 자동 재시도 없음. 아래 가격 예제는 0이므로 그대로 사용하면 차단된다:

```json
{"provider":"bedrock","model_id":"amazon.titan-embed-text-v2:0","region":"ap-southeast-2","verified_at":"","source_url":"https://aws.amazon.com/bedrock/pricing/","input_usd_per_million":0}
```

OpenAI는 provider `openai`, region `global`, 해당 model ID와 공식 `https://openai.com/api/pricing/` 출처를 사용한다. 실제 요금 확인·키 설정·예산 테이블 준비·호출 승인은 이번 작업에서 진행하지 않았다.

## API 호출 없는 명령

```sh
uv run rag cache-info
uv run rag cache-export --dest artifacts/portable-cache.sqlite3
# 기존 캐시에서 FAISS만 다시 구성; miss는 오류이며 API 호출 없음
uv run rag index --data data/scifact --dest artifacts/scifact-rebuilt --config configs/embedding-bedrock.json --cache artifacts/embedding-cache.sqlite3
```

export는 SQLite backup으로 WAL의 완료 batch까지 담는 별도 파일과 SHA256 manifest를 만든다. 파일과 manifest를 함께 보관하고 다른 환경에서 checksum을 확인한 후 `--cache`로 같은 파일을 사용한다. FAISS 파일을 지워도 캐시는 별도로 남는다. 모델·원본·벡터 캐시·API 키는 Git에서 제외한다. AWS에서는 완성한 FAISS index를 S3 immutable version으로 내려받아 재사용하며, 캐시 snapshot은 별도 백업 대상으로 보관할 수 있다. 자동 S3 업로드는 하지 않는다.

## 호출을 승인한 이후의 명령 (이번에는 실행하지 않음)

```sh
uv run rag index --data data/scifact --dest artifacts/scifact-api --config configs/embedding-bedrock.json --allow-api
uv run rag evaluate --data data/scifact --index artifacts/scifact-api --split train --dest results/scifact-api-train --allow-embedding-api
```

배포와 rollback의 인증 검색 smoke도 query embedding API를 호출하므로 별도 `--allow-embedding-api` 옵션이 필요하다. `--generate-smoke`는 Nova 생성까지 추가하는 별도 옵션이다.

## Docker

로컬 API 임베딩·캐시·FAISS 색인은 Python/uv로 실행하므로 Docker가 필요 없다. 현재 AWS Lambda 컨테이너 배포, 로컬 DynamoDB 통합 테스트와 ONNX **리랭커**에는 기존 구성이 남는다. 리랭커는 임베딩 모델과 별개이며 로컬 임베딩 모델은 코드 경로에서 제거했다.

API 응답은 HTTP mock/SDK mock으로만 검증했다. 캐시 재사용·재시작·손상·namespace 분리·FAISS 재구성은 작은 자체 작성 fixture로 검증한다. 실제 API 품질·지연·요금·권한은 호출 전 상태다.
