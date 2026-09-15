'use client';

import { useEffect, useRef } from 'react';
import { marked } from 'marked';
import { X } from 'lucide-react';
import type { ViewerTab, Citation } from '@/lib/types';

marked.setOptions({ breaks: true, gfm: true });

// ─── text utilities ───────────────────────────────────────────────────────────

/** Remove markdown syntax to get plain searchable text */
function stripMarkdown(md: string): string {
  return md
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\*{1,2}([^*\n]+)\*{1,2}/g, '$1')
    .replace(/_{1,2}([^_\n]+)_{1,2}/g, '$1')
    .replace(/`{1,3}([^`\n]+)`{1,3}/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^\s*[-*+>]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/\|[-:\s|]+\|/g, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const escRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// ─── cross-node highlight ────────────────────────────────────────────────────
//
// Strategy: collect every text node with its offset in the joined text
// string, search for the target there, then wrap only the matched nodes.

interface NodeEntry { node: Text; start: number; end: number }

function collectTextNodes(container: HTMLElement): { entries: NodeEntry[]; full: string } {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const entries: NodeEntry[] = [];
  let cursor = 0;
  let n: Node | null;
  while ((n = walker.nextNode())) {
    const text = (n as Text).textContent ?? '';
    entries.push({ node: n as Text, start: cursor, end: cursor + text.length });
    cursor += text.length;
  }
  return { entries, full: entries.map(e => e.node.textContent ?? '').join('') };
}

/**
 * Find `needle` inside `fullText` (flexible whitespace, case-insensitive).
 * Returns [matchStart, matchEnd] in fullText, or null.
 */
function findInFull(fullText: string, needle: string): [number, number] | null {
  // Build a regex that allows any whitespace between words
  const words = needle.split(/\s+/).filter(Boolean);
  if (!words.length) return null;
  const pattern = words.map(escRe).join('[\\s\\S]{0,10}');  // up to 10 chars gap for cross-node whitespace
  const re = new RegExp(pattern, 'i');
  const m = fullText.match(re);
  if (!m || m.index === undefined) return null;
  return [m.index, m.index + m[0].length];
}

/**
 * Wrap the matched region [absStart, absEnd] across however many text nodes
 * it spans. Returns the first <mark> inserted (for scrolling), or null.
 */
function markRange(entries: NodeEntry[], absStart: number, absEnd: number): HTMLElement | null {
  let firstMark: HTMLElement | null = null;

  for (const entry of entries) {
    if (entry.end <= absStart || entry.start >= absEnd) continue;  // no overlap

    const localStart = Math.max(0, absStart - entry.start);
    const localEnd   = Math.min(entry.node.textContent!.length, absEnd - entry.start);
    const text       = entry.node.textContent!;

    const frag = document.createDocumentFragment();
    if (localStart > 0) frag.appendChild(document.createTextNode(text.slice(0, localStart)));
    const mark = document.createElement('mark');
    mark.className = 'hl';
    mark.textContent = text.slice(localStart, localEnd);
    frag.appendChild(mark);
    if (localEnd < text.length) frag.appendChild(document.createTextNode(text.slice(localEnd)));
    entry.node.parentNode?.replaceChild(frag, entry.node);

    if (!firstMark) firstMark = mark;
  }

  return firstMark;
}

/**
 * Main entry: strip markdown from chunk_text, try progressively shorter
 * substrings in the full joined text, place <mark class="hl"> on match.
 * Returns true if at least one mark was placed.
 */
function highlightChunk(container: HTMLElement, chunkText: string, isTable: boolean): boolean {
  // Remove previous highlights so text nodes are restored before we re-collect
  container.querySelectorAll('mark.hl').forEach(m => {
    m.replaceWith(document.createTextNode(m.textContent ?? ''));
  });

  const plain = isTable
    ? (() => {
        for (const line of chunkText.split('\n')) {
          if (line.includes('|') && !/^[\s|:–\-]+$/.test(line)) {
            const cells = line.split('|').map(c => c.trim()).filter(Boolean);
            if (cells.length) return cells[0];
          }
        }
        return stripMarkdown(chunkText);
      })()
    : stripMarkdown(chunkText);

  if (plain.length < 10) return false;

  // Collect text nodes fresh (after removing old marks)
  const { entries, full } = collectTextNodes(container);

  // Try at 3 lengths: full → first 120 chars → first 60 chars
  const attempts = [plain, plain.slice(0, 120), plain.slice(0, 60)]
    .map(s => s.trim())
    .filter((s, i, a) => s.length >= 12 && a.indexOf(s) === i);

  for (const attempt of attempts) {
    const range = findInFull(full, attempt);
    if (!range) continue;
    const [absStart, absEnd] = range;
    const firstMark = markRange(entries, absStart, absEnd);
    if (firstMark) {
      setTimeout(() => firstMark.scrollIntoView({ behavior: 'smooth', block: 'center' }), 80);
      return true;
    }
  }

  return false;
}

// ─── markdown renderer ────────────────────────────────────────────────────────

function parseMarkdown(content: string): string {
  const withBreaks = content.replace(
    /<!--\s*page:(\d+)\s*-->/g,
    '\n\n---\n*— Page $1 —*\n\n'
  );
  return marked.parse(withBreaks) as string;
}

const EXT_ICON: Record<string, string> = { pdf:'📕', docx:'📘', md:'📝', csv:'📊', txt:'📄' };
const getTabIcon = (f: string) => EXT_ICON[f.split('.').pop()?.toLowerCase() ?? ''] ?? '📄';

// ─── component ────────────────────────────────────────────────────────────────

interface Props {
  tabs       : ViewerTab[];
  activeDocId: string | null;
  pendingHL  : Citation | null;
  onActivate : (docId: string) => void;
  onClose    : (docId: string) => void;
}

export function DocumentViewer({ tabs, activeDocId, pendingHL, onActivate, onClose }: Props) {
  const contentRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (!pendingHL || !activeDocId) return;

    const run = () => {
      const el = contentRefs.current.get(activeDocId);
      if (!el) return;

      // Remove previous "not found" banners
      el.querySelectorAll('[data-hl-banner]').forEach(b => b.remove());

      const found = highlightChunk(el, pendingHL.chunk_text, pendingHL.is_table);

      if (!found) {
        const note = document.createElement('div');
        note.setAttribute('data-hl-banner', '1');
        note.style.cssText = [
          'margin:0 0 16px',
          'padding:10px 14px',
          'background:rgba(32,184,205,0.08)',
          'border:1px solid rgba(32,184,205,0.3)',
          'border-radius:10px',
          'font-size:11px',
          'color:#a0a0a0',
          'line-height:1.6',
        ].join(';');
        note.textContent = `Cited passage (p.${pendingHL.page_number}): "${pendingHL.chunk_text.slice(0, 140)}…"`;
        el.prepend(note);
      }
    };

    const id = requestAnimationFrame(run);
    return () => cancelAnimationFrame(id);
  }, [pendingHL, activeDocId]);

  useEffect(() => {
    const ids = new Set(tabs.map(t => t.docId));
    for (const key of contentRefs.current.keys()) {
      if (!ids.has(key)) contentRefs.current.delete(key);
    }
  }, [tabs]);

  const noTabs = tabs.length === 0;

  return (
    <div className="w-[480px] flex-none flex flex-col bg-bg-surface border-l border-border overflow-hidden">

      {/* Tab bar */}
      <div className="flex-none flex items-end bg-bg-base border-b border-border overflow-x-auto min-h-[40px]">
        {noTabs ? (
          <div className="flex items-center w-full justify-center text-[11px] text-text-faint px-4 pb-2.5">
            Click a source to open it here
          </div>
        ) : (
          tabs.map(tab => (
            <div
              key={tab.docId}
              onClick={() => onActivate(tab.docId)}
              className={`viewer-tab ${tab.docId === activeDocId ? 'tab-active' : ''}`}
            >
              <span className="text-sm flex-none">{getTabIcon(tab.fileName)}</span>
              <span className="tab-name" title={tab.fileName}>{tab.fileName}</span>
              <span
                className="tab-close"
                onClick={e => { e.stopPropagation(); onClose(tab.docId); }}
              >
                <X size={9} strokeWidth={2.5} />
              </span>
            </div>
          ))
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto relative">
        {noTabs && (
          <div className="flex flex-col items-center justify-center h-full text-center p-10">
            <div className="w-16 h-16 rounded-2xl bg-bg-elevated border border-border flex items-center justify-center mb-4">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-text-faint">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
            </div>
            <p className="text-sm font-medium text-text-faint">No document open</p>
            <p className="text-[11px] text-text-faint mt-1 max-w-[200px] leading-relaxed">
              Click any source card to open and highlight the cited passage
            </p>
          </div>
        )}

        {tabs.map(tab => (
          <div key={tab.docId} className={tab.docId === activeDocId ? 'block' : 'hidden'}>
            {tab.loading ? (
              <div className="flex flex-col items-center justify-center gap-3 p-10 text-sm text-text-faint">
                <span className="w-5 h-5 border-2 border-border border-t-accent rounded-full animate-spin" />
                Loading document…
              </div>
            ) : (
              <div
                ref={el => {
                  if (el) contentRefs.current.set(tab.docId, el);
                  else contentRefs.current.delete(tab.docId);
                }}
                className="md-body"
                dangerouslySetInnerHTML={{ __html: parseMarkdown(tab.content) }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
