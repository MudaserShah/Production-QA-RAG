'use client';

import { useState } from 'react';
import { ChevronRight, ExternalLink } from 'lucide-react';
import type { Citation } from '@/lib/types';

const EXT_ICON: Record<string, string> = { pdf:'📕', docx:'📘', md:'📝', csv:'📊', txt:'📄' };
const getExt  = (f: string) => f.split('.').pop()?.toLowerCase() ?? '';
const getIcon = (f: string) => EXT_ICON[getExt(f)] ?? '📄';

interface Props {
  citation    : Citation;
  index       : number;
  onOpenSource: (docId: string, citation: Citation) => void;
}

export function CitationCard({ citation: ref, index, onOpenSource }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="citation-card bg-white rounded-xl border border-slate-200 overflow-hidden hover:border-indigo-300 hover:shadow-md transition-all"
      style={{ animationDelay: `${index * 45}ms` }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 bg-slate-50 border-b border-slate-100">
        <span className="text-base flex-none">{getIcon(ref.file_name)}</span>
        <span className="text-[11px] font-semibold text-slate-600 truncate flex-1" title={ref.file_name}>
          {ref.file_name}
        </span>
        {ref.is_table && (
          <span className="flex-none text-[9px] font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 uppercase tracking-wide">
            Table
          </span>
        )}
        <span className="flex-none text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">
          p.{ref.page_number}
        </span>
        <span className="flex-none text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-600">
          {Math.round(ref.relevance_score * 100)}%
        </span>
      </div>

      {/* Expandable excerpt */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-[11px] text-slate-500 hover:text-indigo-600 transition-colors text-left"
      >
        <ChevronRight className={`w-3 h-3 flex-none text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`} />
        Chunk {ref.chunk_index + 1} — {open ? 'collapse' : 'expand to read source'}
      </button>

      {open && (
        <div className="px-4 pb-3">
          <blockquote className="text-[11px] leading-relaxed text-slate-600 border-l-2 border-indigo-200 pl-3 py-1.5 bg-indigo-50 rounded-r-lg whitespace-pre-wrap">
            {ref.chunk_text}
          </blockquote>
        </div>
      )}

      {/* View source button */}
      <div className="px-4 pb-3.5">
        <button
          onClick={() => onOpenSource(ref.doc_id, ref)}
          className="flex items-center gap-1.5 text-[11px] font-semibold text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-lg transition-colors"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          View Source
        </button>
      </div>
    </div>
  );
}
