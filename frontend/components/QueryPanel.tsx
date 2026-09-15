'use client';

import { useState, useRef, useEffect } from 'react';
import { ArrowUp } from 'lucide-react';
import type { Citation } from '@/lib/types';

const EXT_ICON: Record<string, string> = { pdf:'PDF', docx:'DOC', md:'MD', csv:'CSV', txt:'TXT' };
const EXT_COLOR: Record<string, string> = {
  pdf : 'text-rose-400 bg-rose-400/10',
  docx: 'text-blue-400 bg-blue-400/10',
  md  : 'text-emerald-400 bg-emerald-400/10',
  csv : 'text-amber-400 bg-amber-400/10',
  txt : 'text-text-muted bg-bg-elevated',
};
const getExt = (f: string) => f.split('.').pop()?.toLowerCase() ?? '';

interface Props {
  answer      : string;
  streaming   : boolean;
  streamPhase : 'idle' | 'searching' | 'generating';
  citations   : Citation[];
  onQuery     : (question: string, topK: number) => void;
  onOpenSource: (docId: string, citation: Citation) => void;
}

export function QueryPanel({ answer, streaming, streamPhase, citations, onQuery, onOpenSource }: Props) {
  const [question, setQuestion] = useState('');
  const [topK, setTopK]         = useState(5);
  const bottomRef               = useRef<HTMLDivElement>(null);
  const textareaRef             = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (answer && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [answer]);

  function submit() {
    if (!question.trim() || streaming) return;
    onQuery(question.trim(), topK);
  }

  const hasContent = answer || citations.length > 0 || streaming;

  return (
    <main className="flex-1 min-w-0 flex flex-col bg-bg-base overflow-hidden border-r border-border">

      {/* Scrollable results area */}
      <div className="flex-1 overflow-y-auto">
        {!hasContent ? (
          /* Empty state */
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <div className="w-14 h-14 rounded-2xl bg-bg-surface border border-border flex items-center justify-center mb-5">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-text-faint">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
            </div>
            <p className="text-sm font-semibold text-text-muted mb-1">Ask anything</p>
            <p className="text-[12px] text-text-faint max-w-xs leading-relaxed">
              Upload documents to your library, then ask questions to get AI-powered answers with source citations.
            </p>
          </div>
        ) : (
          <div className="px-6 py-6 space-y-6 max-w-3xl mx-auto w-full">

            {/* Streaming / searching indicator */}
            {streaming && streamPhase === 'searching' && (
              <div className="flex items-center gap-3 slide-up">
                <div className="flex gap-1">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
                <span className="text-[12px] text-text-muted">Searching documents…</span>
              </div>
            )}

            {/* Source cards strip */}
            {citations.length > 0 && (
              <div className="slide-up">
                <p className="text-[10px] font-semibold text-text-faint uppercase tracking-widest mb-3">
                  Sources · {citations.length}
                </p>
                <div className="flex gap-2 overflow-x-auto pb-1">
                  {citations.map((ref, i) => {
                    const ext = getExt(ref.file_name);
                    const colorCls = EXT_COLOR[ext] ?? 'text-text-muted bg-bg-elevated';
                    return (
                      <button
                        key={`${ref.doc_id}-${ref.chunk_index}`}
                        onClick={() => onOpenSource(ref.doc_id, ref)}
                        className="flex-none flex items-center gap-2 bg-bg-surface border border-border hover:border-border-accent hover:bg-accent-dim rounded-xl px-3 py-2.5 transition-all group"
                        style={{ animationDelay: `${i * 40}ms` }}
                        title={ref.chunk_text.slice(0, 200)}
                      >
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-md flex-none ${colorCls}`}>
                          {EXT_ICON[ext] ?? 'FILE'}
                        </span>
                        <div className="text-left min-w-0">
                          <p className="text-[11px] font-medium text-text-primary truncate max-w-[120px] group-hover:text-accent transition-colors">
                            {ref.file_name}
                          </p>
                          <p className="text-[9px] text-text-faint">
                            p.{ref.page_number} · {Math.round(ref.relevance_score * 100)}% match
                          </p>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Generating indicator */}
            {streaming && streamPhase === 'generating' && !answer && (
              <div className="flex items-center gap-3">
                <div className="flex gap-1">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
                <span className="text-[12px] text-text-muted">Generating answer…</span>
              </div>
            )}

            {/* Answer */}
            {(answer || (streaming && streamPhase === 'generating')) && (
              <div className="slide-up">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-5 h-5 rounded-md bg-accent-dim border border-border-accent flex items-center justify-center">
                    <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" className="text-accent">
                      <path d="M8 1.5a.5.5 0 0 1 .5.5v5.5H14a.5.5 0 0 1 0 1H8.5V14a.5.5 0 0 1-1 0V8.5H2a.5.5 0 0 1 0-1h5.5V2a.5.5 0 0 1 .5-.5z"/>
                    </svg>
                  </div>
                  <span className="text-[10px] font-semibold text-text-faint uppercase tracking-widest">Answer</span>
                </div>
                <div className="text-[13.5px] leading-[1.85] text-text-primary whitespace-pre-wrap font-sans">
                  {answer}
                  {streaming && <span className="cursor" />}
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="flex-none border-t border-border bg-bg-surface p-4">
        <div className="max-w-3xl mx-auto">
          <div className="relative bg-bg-elevated border border-border rounded-2xl focus-within:border-border-accent transition-colors">
            <textarea
              ref={textareaRef}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
              }}
              rows={2}
              placeholder="Ask anything about your documents…"
              className="w-full bg-transparent px-4 py-3.5 pr-24 text-[13px] text-text-primary placeholder:text-text-faint focus:outline-none resize-none leading-relaxed"
            />
            <div className="absolute right-3 bottom-3 flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-text-faint">k=</span>
                <input
                  type="number"
                  value={topK}
                  min={1} max={20}
                  onChange={e => setTopK(parseInt(e.target.value) || 5)}
                  className="w-10 bg-bg-surface border border-border rounded-lg text-[11px] text-text-muted text-center py-1 focus:outline-none focus:border-border-accent"
                />
              </div>
              <button
                onClick={submit}
                disabled={streaming || !question.trim()}
                className="w-8 h-8 rounded-full bg-accent hover:bg-accent-hover active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all shadow-sm"
              >
                {streaming ? (
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <ArrowUp className="w-4 h-4 text-white" strokeWidth={2.5} />
                )}
              </button>
            </div>
          </div>
          <p className="text-center text-[10px] text-text-faint mt-2">
            Press <kbd className="px-1.5 py-0.5 rounded bg-bg-elevated border border-border text-[9px]">Enter</kbd> to send · <kbd className="px-1.5 py-0.5 rounded bg-bg-elevated border border-border text-[9px]">Shift+Enter</kbd> for new line
          </p>
        </div>
      </div>
    </main>
  );
}
