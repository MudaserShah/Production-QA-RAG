export interface DocRecord {
  doc_id: string;
  file_name: string;
  file_type: string;
  total_chunks: number;
  total_pages: number;
  upload_timestamp: string;
  status: string;
}

export interface Citation {
  file_name: string;
  page_number: number;
  chunk_index: number;
  chunk_text: string;
  relevance_score: number;
  doc_id: string;
  is_table: boolean;
}

export interface ViewerTab {
  docId: string;
  fileName: string;
  fileType: string;
  content: string;   // full markdown; empty string while loading
  loading: boolean;
}
