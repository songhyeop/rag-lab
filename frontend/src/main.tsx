import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Result = {
  answer?: string;
  abstained?: boolean;
  citations?: string[];
  request_id: string;
  index_version: string;
  dataset: string;
  hits: {
    id: string;
    doc_id: string;
    title: string;
    text: string;
    score: number;
  }[];
  timings: Record<string, number>;
  usage?: Record<string, number>;
};
declare global {
  interface Window {
    RAG_CONFIG: { googleClientId: string; apiBase: string };
    google?: {
      accounts: {
        id: {
          initialize: (x: {
            client_id: string;
            callback: (x: { credential: string }) => void;
          }) => void;
          renderButton: (
            el: HTMLElement,
            options: Record<string, unknown>,
          ) => void;
          disableAutoSelect: () => void;
        };
      };
    };
  }
}
const config = window.RAG_CONFIG.googleClientId
  ? window.RAG_CONFIG
  : {
      googleClientId: import.meta.env.VITE_GOOGLE_CLIENT_ID || "",
      apiBase: import.meta.env.VITE_API_BASE || "",
    };
function App() {
  const [token, setToken] = useState("");
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState("hybrid_rerank");
  const [result, setResult] = useState<Result>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [usage, setUsage] = useState<unknown>();
  const [experiments, setExperiments] = useState<unknown>();
  const button = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (token || !config.googleClientId) return;
    const interval = window.setInterval(() => {
      if (!window.google || !button.current || !config.googleClientId) return;
      window.google.accounts.id.initialize({
        client_id: config.googleClientId,
        callback: ({ credential }) => {
          setToken(credential);
        },
      });
      window.google.accounts.id.renderButton(button.current, {
        theme: "outline",
        size: "large",
        locale: "ko",
      });
      clearInterval(interval);
    }, 200);
    return () => clearInterval(interval);
  }, [token]);
  useEffect(() => {
    setResult(undefined);
    setUsage(undefined);
    setExperiments(undefined);
    if (!token) return;
    // Used for UI expiry only; the API verifies signatures and claims independently.
    const payload = JSON.parse(
      atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    );
    const timeout = window.setTimeout(
      () => setToken(""),
      Math.max(0, payload.exp * 1000 - Date.now()),
    );
    return () => clearTimeout(timeout);
  }, [token]);
  async function api(path: string, body?: unknown) {
    const response = await fetch(config.apiBase + path, {
      method: body ? "POST" : "GET",
      headers: {
        Authorization: `Bearer ${token}`,
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (response.status === 401) setToken("");
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `요청 실패 (${response.status})`);
    }
    return response.json();
  }
  async function run(kind: "search" | "answer") {
    setBusy(true);
    setError("");
    setResult(undefined);
    try {
      setResult(await api("/" + kind, { question, method }));
      setUsage(await api("/usage"));
    } catch (e) {
      setError(e instanceof Error ? e.message : "요청을 처리하지 못했습니다.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <main>
      <header>
        <a href="/" className="brand">
          RAG <span>LAB</span>
          <b>실험 노트 01</b>
        </a>
        {token && (
          <button
            className="quiet"
            onClick={() => {
              setToken("");
              setResult(undefined);
              setUsage(undefined);
              setExperiments(undefined);
              window.google?.accounts.id.disableAutoSelect();
            }}
          >
            로그아웃
          </button>
        )}
      </header>
      <section className="intro">
        <div className="eyebrow">RETRIEVAL · EVIDENCE · EVALUATION</div>
        <h1>
          좋은 답변은
          <br />
          <em>근거에서</em> 시작됩니다.
        </h1>
        <p>
          검색 방식을 바꾸고, 같은 질문의 근거를 비교하세요.
          <br />
          공개 데이터로 살펴보는 작은 RAG 실험실입니다.
        </p>
      </section>
      {!token ? (
        <section className="login">
          <h2>초대받은 계정으로 시작하세요</h2>
          <p>Google 계정 최대 5명에게 열린 연구용 데모입니다.</p>
          <div ref={button} />
          {!config.googleClientId && (
            <p className="notice">Google 로그인 설정을 준비 중입니다.</p>
          )}
        </section>
      ) : (
        <>
          <section className="workspace">
            <div className="section-title">
              <span>01 / 질문과 검색</span>
              <small>질문·답변은 영어로</small>
            </div>
            <label htmlFor="question">어떤 근거를 찾고 있나요?</label>
            <textarea
              id="question"
              value={question}
              maxLength={2000}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question in English…"
            />
            <div className="controls">
              <select
                aria-label="검색 방식"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
              >
                <option value="bm25">BM25 · 키워드</option>
                <option value="dense">Dense · 의미 검색</option>
                <option value="hybrid">Hybrid · 결합 검색</option>
                <option value="hybrid_rerank">Hybrid + Reranker</option>
              </select>
              <button
                disabled={busy || !question.trim()}
                onClick={() => run("search")}
              >
                근거 검색
              </button>
              <button
                className="primary"
                disabled={busy || !question.trim()}
                onClick={() => run("answer")}
              >
                {busy ? "확인 중…" : "답변 생성 ↗"}
              </button>
            </div>
            <p className="fine">
              생성 30회/일 · 처리 중에는 잠시 기다려 주세요. 근거가 부족하면
              답변을 유보합니다.
            </p>
          </section>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
          {result && (
            <section className="results" aria-live="polite">
              <div className="section-title">02 / 답변과 근거</div>
              {result.answer && (
                <div className="answer">
                  <small>
                    {result.abstained ? "근거 부족 · 답변 유보" : "답변"}
                  </small>
                  <h2>{result.answer}</h2>
                  <p>
                    인용:{" "}
                    {result.citations
                      ?.map(
                        (id) =>
                          result.hits.find((h) => h.doc_id === id)?.title || id,
                      )
                      .join(", ") || "없음"}
                  </p>
                </div>
              )}
              {result.hits.slice(0, 10).map((hit, i) => (
                <details key={hit.id}>
                  <summary>
                    <span>{String(i + 1).padStart(2, "0")}</span> {hit.title}
                  </summary>
                  <p>{hit.text}</p>
                  <small>{hit.doc_id}</small>
                </details>
              ))}
              <p className="fine">
                {result.dataset} · 검색 {result.timings.total_ms.toFixed(0)} ms
                · 인덱스 {result.index_version.slice(0, 12)} · 요청{" "}
                {result.request_id}
              </p>
            </section>
          )}
          <section className="meta">
            <button
              onClick={async () => {
                try {
                  setUsage(await api("/usage"));
                } catch (e) {
                  setError(String(e));
                }
              }}
            >
              사용량 확인
            </button>
            <button
              onClick={async () => {
                try {
                  setExperiments(await api("/experiments"));
                } catch (e) {
                  setError(String(e));
                }
              }}
            >
              실험 리포트
            </button>
            {usage !== undefined && (
              <details>
                <summary>사용량 (nano-USD, UTC 일 기준)</summary>
                <pre>{JSON.stringify(usage, null, 2)}</pre>
              </details>
            )}
            {experiments !== undefined && (
              <details open>
                <summary>사전 계산 실험</summary>
                <pre>{JSON.stringify(experiments, null, 2)}</pre>
              </details>
            )}
          </section>
        </>
      )}
      <footer>
        <span>RAG Lab — 재현 가능한 검색 실험</span>
        <p>
          HotpotQA / Wikipedia · CC BY-SA 4.0
          <br />
          축소 코퍼스 실험이며 공식 fullwiki 결과와 직접 비교할 수 없습니다.
        </p>
      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
