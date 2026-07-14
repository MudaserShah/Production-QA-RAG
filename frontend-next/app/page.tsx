'use client';

import { useState, useCallback, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { QueryPanel } from '@/components/QueryPanel';
import { DocumentViewer } from '@/components/DocumentViewer';
import type { DocRecord, Citation, ViewerTab } from '@/lib/types';

const API = 'http://localhost:8001';

export default function Home() {
  const [docs, setDocs]               = useState<DocRecord[]>([]);
  const [citations, setCitations]     = useState<Citation[]>([]);
  const [answer, setAnswer]           = useState('');
  const [streaming, setStreaming]     = useState(false);
  const [streamPhase, setStreamPhase] = useState<'idle'|'searching'|'generating'>('idle');
  const [tabs, setTabs]               = useState<ViewerTab[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [pendingHL, setPendingHL]     = useState<Citation | null>(null);
  const [toastMsg, setToastMsg]       = useState('');
  const [backendOk, setBackendOk]     = useState<boolean | null>(null);

  useEffect(() => {
    if (!toastMsg) return;
    const t = setTimeout(() => setToastMsg(''), 3500);
    return () => clearTimeout(t);
  }, [toastMsg]);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => setBackendOk(r.ok))
      .catch(() => setBackendOk(false));
  }, []);

  const loadDocs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/documents`);
      const d = await r.json();
      setDocs(d.documents ?? []);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const handleQuery = useCallback(async (question: string, topK: number) => {
    setAnswer('');
    setCitations([]);
    setStreaming(true);
    setStreamPhase('searching');

    try {
      const res = await fetch(`${API}/query/stream`, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ question, top_k: topK }),
      });

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let   buf     = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === 'citations') {
              setCitations(ev.data);
              setStreamPhase('generating');
            } else if (ev.type === 'token') {
              setAnswer(prev => prev + ev.content);
            }
          } catch { /* malformed — skip */ }
        }
      }
    } catch (e: unknown) {
      setToastMsg(`Query error: ${e instanceof Error ? e.message : 'unknown'}`);
    } finally {
      setStreaming(false);
      setStreamPhase('idle');
    }
  }, []);

  const handleOpenSource = useCallback(async (docId: string, citation: Citation) => {
    const existing = tabs.find(t => t.docId === docId);

    if (existing && !existing.loading) {
      setActiveDocId(docId);
      setPendingHL(citation);
      return;
    }

    if (!existing) {
      setTabs(prev => [...prev, {
        docId,
        fileName: citation.file_name,
        fileType: '',
        content : '',
        loading : true,
      }]);
    }
    setActiveDocId(docId);

    try {
      const r    = await fetch(`${API}/files/${docId}/markdown`);
      const data = await r.json();
      setTabs(prev => prev.map(t =>
        t.docId === docId
          ? { ...t, content: data.content, fileType: data.file_type, loading: false }
          : t
      ));
      setPendingHL(citation);
    } catch {
      setTabs(prev => prev.filter(t => t.docId !== docId));
      setToastMsg('Could not load document.');
    }
  }, [tabs]);

  const handleCloseTab = useCallback((docId: string) => {
    setTabs(prev => {
      const next = prev.filter(t => t.docId !== docId);
      if (activeDocId === docId) setActiveDocId(next.at(-1)?.docId ?? null);
      return next;
    });
    if (pendingHL && citations.find(c => c.doc_id === docId)) setPendingHL(null);
  }, [activeDocId, pendingHL, citations]);

  const handleDeleteDoc = useCallback(async (docId: string, fileName: string) => {
    if (!confirm(`Delete "${fileName}" and all its indexed data?`)) return;
    await fetch(`${API}/documents/${docId}`, { method: 'DELETE' });
    handleCloseTab(docId);
    await loadDocs();
    setToastMsg(`Deleted "${fileName}"`);
  }, [loadDocs, handleCloseTab]);

  return (
    <div className="h-full flex flex-col bg-bg-base">

      {/* Header */}
      <header className="flex-none h-12 border-b border-border flex items-center px-5 gap-4 bg-bg-surface z-20">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-accent flex items-center justify-center">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10 9 9 9 8 9"/>
            </svg>
          </div>
          <span className="text-sm font-semibold text-text-primary tracking-tight">Document Butler</span>
          <span className="text-[10px] font-medium text-accent bg-accent-dim px-2 py-0.5 rounded-full border border-border-accent">
            Lecture 18
          </span>
        </div>

        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${
              backendOk === null ? 'bg-text-faint' :
              backendOk           ? 'bg-emerald-400' : 'bg-rose-400'
            }`} />
            <span className="text-[11px] text-text-muted">
              {backendOk === null ? 'Connecting…' : backendOk ? 'Connected' : 'Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Three-panel body */}
      <div className="flex flex-1 min-h-0">
        <Sidebar
          docs={docs}
          onUploaded={loadDocs}
          onDelete={handleDeleteDoc}
          onToast={setToastMsg}
        />

        <QueryPanel
          answer={answer}
          streaming={streaming}
          streamPhase={streamPhase}
          citations={citations}
          onQuery={handleQuery}
          onOpenSource={handleOpenSource}
        />

        <DocumentViewer
          tabs={tabs}
          activeDocId={activeDocId}
          pendingHL={pendingHL}
          onActivate={setActiveDocId}
          onClose={handleCloseTab}
        />
      </div>

      {/* Toast */}
      {toastMsg && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-bg-elevated border border-border text-text-primary text-xs font-medium px-5 py-2.5 rounded-full shadow-xl z-50 pointer-events-none fade-in">
          {toastMsg}
        </div>
      )}
    </div>
  );
}
