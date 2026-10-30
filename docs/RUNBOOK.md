# 실행·실험·배포 절차

## 설치와 로컬 검색

```sh
uv sync --frozen --extra data
cd frontend
npm ci
npm run build
cd ..
uv run rag models
uv run rag smoke-models
uv run rag prepare scifact
uv run rag index --data data/scifact --dest artifacts/scifact-256
uv run rag evaluate --data data/scifact --index artifacts/scifact-256 --split train --dest results/scifact-train
```

`configs/models.lock.json`의 upstream commit만 다운로드한다. 모델 artifact.json과 인덱스 manifest를 로딩 시 검증한다. 모델 파일을 수정하거나 새 revision으로 바꾸면 인덱스를 다시 만든다.

처음 준비한 manifest는 `configs/manifests/<dataset>.json`에 보관한다. 다음 준비 시 같은 sources/checksum을 확인해야 하며 변경은 새 실험이다. SciFact test 300개와 HotpotQA dev 200개를 튜닝에 사용하지 않는다.

```sh
uv run rag prepare hotpotqa
uv run rag index --data data/hotpotqa --dest artifacts/hotpotqa-256
uv run rag evaluate --data data/hotpotqa --index artifacts/hotpotqa-256 --split train --dest results/hotpotqa-train
```

HotpotQA 원본 서버가 실패하면 `hotpotqa/hotpot_qa` 게시자 namespace의 고정 revision Parquet를 사용한다. 이 경우 manifest에 대체 URL·revision·SHA256과 사유를 남긴다. 선택 질문 200+200개의 context를 모두 포함하고 train context로 20,000문서를 채운다.

128/256/384 청킹은 `configs/chunk-*.json`으로 각각 인덱스를 만들고 **train**만 평가한다. 설정을 고정한 후에만 `--split test` (SciFact), `--split dev` (HotpotQA)로 최종 평가한다. 최종 평가를 보고 설정을 다시 선택하면 별도 탐색 결과로 표시한다.

결과는 ignored `results/<run>/`에 JSONL·CSV·Markdown, provenance와 수동 검토용 30건을 저장한다. 작은 aggregate report와 provenance만 `docs/reports/`에 복사해 공개한다. 수동 검토 파일의 reviewed가 false면 실패 분석 완료로 주장하지 않는다.

## 인증과 생성

`.env.example`을 `.env`로 복사하고 Google 웹 OAuth client ID, 초대 이메일 JSON 배열(1~5명), 정확한 index version을 설정한다. Google Console의 Authorized JavaScript Origins에 로컬 `http://localhost:5173`와 실제 CloudFront HTTPS origin을 등록한다. 클라이언트 secret은 GIS ID token 방식에서 필요하지 않다. 초대 목록은 공개 파일에 쓰지 않는다.

```sh
uv run uvicorn rag_lab.api:app --port 8000
cd frontend
npm run dev
```

로컬 화면은 `frontend/.env.example`을 `frontend/.env.local`로 복사하고 `VITE_GOOGLE_CLIENT_ID`를 설정한다. API base는 `http://localhost:8000`이다. `.env.local`은 Git에서 제외한다. 배포에서는 CI 산출물에 runtime-config.js를 주입하므로 실제 OAuth 설정 없이 후보를 빌드한다. 토큰은 React 메모리에만 있고 만료되면 다시 로그인한다.

생성에는 실제 DynamoDB 테이블과 공식 시드니 단가를 확인한 `configs/pricing.local.json`이 필요하다. `uv run python -m scripts.freeze_pricing`는 공식 AWS Price List의 시드니 On-demand Nova Lite SKU만 추출하고 단가와 증거를 함께 고정한다. 파일 형식은 `model_id`, `region`, `input_usd_per_million`, `output_usd_per_million`, `verified_at`, `source_url`이다. 원본 요금표 증거도 함께 보관한다. 미설정은 503으로 차단한다. AWS SDK의 기존 `aws login` 자격 증명은 boto3[crt]로 사용하며 키나 세션을 복사하지 않는다.

로컬 생성 평가는 명시적으로만 실행한다:

```sh
uv run rag evaluate --data data/hotpotqa --index artifacts/hotpotqa-256 --split dev --dest results/hotpotqa-generated --generate --subject YOUR_GOOGLE_SUB
```

데모는 사용자별 30회/UTC일이다. 명시적 로컬 생성 평가는 웹의 일일 30회 제한을 적용하지 않으며 같은 DynamoDB의 $30 총예산과 동시성 제한은 유지한다. 평가와 데모 사용량은 동일한 subject로 합산된다. 전체 비용은 입력/출력 token과 request ledger에 남는다. Nova Lite는 실제 CountTokens 호출에서 지원되지 않음을 확인했으므로 호출 전 모델 최대 입력 300,000토큰과 출력 512토큰 비용을 예약한다. 전체 payload는 32KiB로 제한하며 실제 응답 usage로 차액을 정산한다. SDK 재시도는 0회로, 최대 1회라는 계획 상한 내다. 실패 후 사용량 불명은 최대 예약을 유지한다. 프로세스 종료로 active lock이 남으면 Bedrock 요청 결과를 확인해 수동 정산해야 한다. 자동 TTL로 비용을 환불하지 않는다.

## CI와 로컬 CD

PR `backend`, `frontend`가 필수 검사다. main push는 같은 검사를 재사용해 amd64 Lambda 이미지를 만들고 실제 ONNX 검색과 Lambda runtime의 health/auth 차단을 검증한다. 이후 image.tar.gz, frontend.tar.gz, checks.json, candidate.json을 `candidate` artifact로 3일 보관한다. CI에는 AWS 자격 증명·OAuth 실제 설정이 없다.

```sh
uv run python -m scripts.candidate --run-id CI_RUN_ID --dest artifacts/selected
```

성공한 이 저장소 main의 push workflow인지, commit과 run ID, 모든 SHA256을 검사한다. `latest`나 로컬 재빌드 이미지는 배포하지 않는다.

## 인프라 변경: 반드시 change set 검토 후 적용

부트스트랩은 ECR 저장소·비공개 versioned index S3만 만든다:

```sh
uv run python -m scripts.infra plan --stack rag-lab-storage --template infra/bootstrap.yaml
uv run python -m scripts.infra apply --change-set EXACT_CHANGE_SET_ARN --expected-account EXPECTED_ACCOUNT
```

plan 결과를 검토하고 별도 적용 확인을 받은 뒤 apply한다. Free 플랜·IAM·SAM·공개 endpoint가 거부되면 정확한 오류를 기록하고 중단한다. 정책·플랜 전환 금지.

부트스트랩 적용 후 아래 명령으로 ECR에 **CI 이미지**를 그대로 옮기고 인덱스 zip(manifest.json, chunks.jsonl, vectors.faiss) 및 S3 VersionId를 고정한다.

```sh
uv run python -m scripts.stage --run-id CI_RUN_ID --expected-account EXPECTED_ACCOUNT --index artifacts/hotpotqa-256
```

`configs/staged.local.json`에 기록된 image digest·index 값에 Google·단가·알림 설정을 더해 service parameters를 완성한다. 서비스 template은 별도 change set으로 검토한다. `configs/service-parameters.example.json`을 복사해 지역·이미지 digest·Google·가격·알림 email을 채운 local 파일을 전달한다. AlertEmail은 SNS 구독 이메일 확인이 필요하다. 알림은 비용 차단이 아니다. 운영 알림은 크레딧/환불을 제외한 계정 내 지정 서비스의 실제 비용 합계다. BudgetStartEpoch·BudgetEndEpoch로 배포부터 Free 플랜 만료 전까지의 일회성 기간을 고정해 자동 갱신을 피한다. 기존의 다른 프로젝트 서비스 비용도 포함하므로 RAG 전용 비용으로 해석하지 않는다.

```sh
uv run python -m scripts.infra plan --stack rag-lab --template infra/template.yaml --parameters configs/service-parameters.local.json
uv run python -m scripts.infra apply --change-set EXACT_CHANGE_SET_ARN --expected-account EXPECTED_ACCOUNT
```

애플리케이션 배포는 `configs/deploy.example.json`을 local 파일로 복사하고 현재 account·index version·S3 object version·SHA·OAuth를 고정한다. `RAG_SMOKE_GOOGLE_TOKEN`에 메모리에서 복사한 유효 초대 Google ID token을 설정한다. 토큰은 파일·Git·shell history에 남기지 않는다.

```sh
uv run python -m scripts.deploy --run-id CI_RUN_ID --config configs/deploy.local.json
# 실제 생성 호출까지 검증할 때만 --generate-smoke 추가 (공유 예산에 포함)
uv run python -m scripts.rollback --release releases/CI_RUN_ID.json
```

배포는 candidate Lambda 버전의 health·비인증 차단·인증 검색·index version을 확인하고 alias를 전환한다. 프런트는 S3 release 경로별로 저장하며 CDN origin 경로를 전환한다. post-smoke 실패 시 이전 alias·frontend로 복원한다. 호환되지 않는 API 변경은 같은 릴리스에서도 전환 간격이 있으므로 이전/새 UI 양쪽을 지원하는 과도기 API가 필요하다.

release manifest에는 이전 version의 환경(index/가격/OAuth), frontend path, 새 ECR digest·commit·CI run이 남는다. private ignored 파일로 취급한다. rollback은 이전 불변 Lambda version의 인덱스와 프런트를 복원하고 실제 검색 smoke를 수행한다. 실제 rollback을 실행하기 전에는 검증 완료로 표시하지 않는다.

## 검증

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
sam validate --lint --template infra/template.yaml --region ap-southeast-2
```

PR에서는 고정 벡터·자체 문서 fixture만 사용한다. 실제 ONNX smoke는 `uv run rag smoke-models`이며 GitHub 후보 이미지에서도 실행한다. 모든 업무 경로의 401을 모델 로딩 전 검증하고, 원자 예산·동시성·일일 제한·잘못된 citation·CI 출처·checksum·AWS account/index 누락을 테스트한다.

병렬 예산 검사는 Moto가 스레드 안전하지 않아 DynamoDB Local에서 수행한다. CI의 backend job도 같은 pinned image를 사용하며 실제 AWS 인증은 없다.

```sh
docker run -d --name rag-lab-dynamodb -p 127.0.0.1:8001:8000 amazon/dynamodb-local@sha256:6ada631ec7aa5fc83446855eb17233e3dad184fec47477e8140c6c569cfe0c86
RAG_TEST_DYNAMODB_ENDPOINT=http://127.0.0.1:8001 uv run pytest -q
```

리포트 검증은 원본 문서·질의 ID 연결과 평가값 재계산을 포함한다:

```sh
uv run rag validate-report results/scifact-train --data data/scifact
```
