import type { Metadata } from 'next';
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import './globals.css';

export const metadata: Metadata = {
  title: 'Document Butler — AI Document Analysis',
  description: 'Upload documents, ask questions, see exact source citations.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} h-full`}>
      <body className="h-full overflow-hidden bg-bg-base text-text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
