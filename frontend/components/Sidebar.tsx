'use client';

import { useRef } from 'react';
import { Upload, Trash2, FileText } from 'lucide-react';
import type { DocRecord } from '@/lib/types';

const API = 'http://localhost:8001';

const EXT_COLOR: Record<string, string> = {
  pdf : 'text-rose-400',
  docx: 'text-blue-400',
  md  : 'text-emerald-400',
  csv : 'text-amber-400',
  txt : 'text-text-muted',
};
const SUPPORTED = new Set(['pdf','docx','md','csv','txt']);
const getExt    = (f: string) => f.split('.').pop()?.toLowerCase() ?? '';

const FileIcon = ({ name }: { name: string }) => {
  const ext = getExt(name);
  const cls = EXT_COLOR[ext] ?? 'text-text-muted';
  return (
    <div className={`w-7 h-7 rounded-md bg-bg-elevated flex items-center justify-center flex-none ${cls}`}>
      <span className="text-[10px] font-bold uppercase">{ext.slice(0,3)}</span>
    </div>
  );
};

interface Props {
  docs      : DocRecord[];
  onUploaded: () => void;
  onDelete  : (docId: string, fileName: string) => void;
  onToast   : (msg: string) => void;
}

export function Sidebar({ docs, onUploaded, onDelete, onToast }: Props) {
  const zoneRef  = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function upload(files: File[]) {
    const valid   = files.filter(f => SUPPORTED.has(getExt(f.name)));
    const invalid = files.length - valid.length;
    if (invalid) onToast(`Skipped ${invalid} unsupported file(s)`);
    if (!valid.length) return;

    zoneRef.current?.classList.add('drop-uploading');
    onToast(`Indexing ${valid.length} file(s)…`);

    const fd = new FormData();
    valid.forEach(f => fd.append('files', f));

    try {
      const r    = await fetch(`${API}/upload`, { method: 'POST', body: fd });
      const data = await r.json();
      onToast(data.failed > 0
        ? `${data.successful} indexed · ${data.failed} failed`
        : `${data.successful} file(s) ready`);
      onUploaded();
    } catch (e: unknown) {
      onToast(`Upload error: ${e instanceof Error ? e.message : 'unknown'}`);
    } finally {
      zoneRef.current?.classList.remove('drop-uploading');
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    zoneRef.current?.classList.remove('drop-active');
    upload(Array.from(e.dataTransfer.files));
  }

  return (
    <aside className="w-60 flex-none bg-bg-surface border-r border-border flex flex-col overflow-hidden">

      {/* Header */}
      <div className="px-4 pt-4 pb-3">
        <p className="text-[10px] font-semibold text-text-faint uppercase tracking-widest mb-3">Library</p>

        {/* Drop zone */}
        <div
          ref={zoneRef}
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); zoneRef.current?.classList.add('drop-active'); }}
          onDragLeave={e => { if (!zoneRef.current?.contains(e.relatedTarget as Node)) zoneRef.current?.classList.remove('drop-active'); }}
          onDrop={onDrop}
          className="border border-dashed border-border rounded-xl p-4 text-center cursor-pointer hover:border-accent hover:bg-accent-dim transition-all duration-200 group"
        >
          <div className="w-8 h-8 mx-auto mb-2 rounded-lg bg-bg-elevated flex items-center justify-center group-hover:bg-accent-dim transition-colors">
            <Upload className="w-4 h-4 text-text-faint group-hover:text-accent transition-colors" />
          </div>
          <p className="text-[11px] font-medium text-text-muted group-hover:text-text-primary transition-colors">Drop files here</p>
          <p className="text-[10px] text-text-faint mt-0.5">
            or <span className="text-accent font-semibold">browse</span>
          </p>
          <p className="text-[9px] text-text-faint mt-2 tracking-wide">PDF · DOCX · MD · CSV · TXT</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept=".pdf,.docx,.md,.csv,.txt"
          multiple
          onChange={e => { if (e.target.files?.length) upload(Array.from(e.target.files)); e.target.value = ''; }}
        />
      </div>

      {/* Divider */}
      <div className="h-px bg-border mx-4" />

      {/* Document list */}
      <div className="flex-1 overflow-y-auto py-2 px-2">
        {docs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <FileText className="w-6 h-6 text-text-faint mb-2" />
            <p className="text-[11px] text-text-faint">No documents yet</p>
          </div>
        ) : docs.map(doc => (
          <div
            key={doc.doc_id}
            className="doc-row group flex items-center gap-2.5 px-2 py-2 rounded-lg hover:bg-bg-elevated transition-colors mb-0.5 cursor-default"
          >
            <FileIcon name={doc.file_name} />
            <div className="flex-1 min-w-0">
              <p className="text-[11px] font-medium text-text-primary truncate leading-tight" title={doc.file_name}>
                {doc.file_name}
              </p>
              <p className="text-[9px] text-text-faint mt-0.5">
                {doc.total_pages > 0 && `${doc.total_pages}p`}
                {doc.total_chunks > 0 && ` · ${doc.total_chunks} chunks`}
              </p>
            </div>
            <button
              onClick={() => onDelete(doc.doc_id, doc.file_name)}
              className="del-btn flex-none p-1 rounded-md text-text-faint hover:text-rose-400 hover:bg-rose-400/10 transition-all"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {/* Footer */}
      {docs.length > 0 && (
        <div className="px-4 py-3 border-t border-border">
          <p className="text-[10px] text-text-faint">{docs.length} document{docs.length !== 1 && 's'} indexed</p>
        </div>
      )}
    </aside>
  );
}
