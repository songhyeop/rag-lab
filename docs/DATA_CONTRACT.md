# 데이터·평가 계약

`data/<dataset>/manifest.json`은 schema_version, dataset, seed, sources(URL/SHA256/라이선스/split), corpus_sha256, labels_sha256, counts, document_ids_sha256을 포함한다. labels 파일은 train/test 또는 train/dev 파일로 분리한다. corpus.jsonl 각 행은 `id,title,text`만 허용한다. labels 행은 `id,question,answer,relevant`이고 relevant는 doc ID -> relevance grade 사전이다.

HotpotQA 문서 ID는 정규화한 제목과 본문의 SHA256이다. 같은 제목·다른 본문은 별개 문서로 유지하며 gold는 해당 질문 context의 정확한 ID에 연결한다. 샘플링은 정렬한 질문 ID를 seed 42로 shuffle하여 앞 200개다. 선택된 context를 먼저 모두 포함하고 train context로 정확히 20,000개까지 보충한다. 같은 질문 ID가 train/dev에 중복되면 실패한다.

인덱스 manifest는 원본 manifest hash, 청킹 설정, embedding/reranker repository revision, 모델 파일 SHA256, 청크·벡터 파일 SHA256을 포함한다. 아티팩트 hash는 다운로드와 로딩 시 검증한다. Git에는 자체 작성 fixture만 둔다. HotpotQA/Wikipedia의 CC BY-SA 4.0 출처 표시와 SciFact의 초록/주석 권리 구분은 THIRD_PARTY_NOTICES.md를 참고한다.
