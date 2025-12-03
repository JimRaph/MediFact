import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Providers from '@/providers/SessionProviders'
import QueryProvider from '@/providers/QueryProvider'
import { ToastContainer } from "react-toastify";


const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: 'MediFact - Your hub for medical information',
    template: '%s | MediFact - Accurate WHO Health Insights'
  },
  description: `MediFact: AI-powered retrieval from WHO fact sheets on diseases, vaccines,
   global health risks, and public health data. Get precise, authoritative answers using vector 
   search and RAG technology.`,
   icons: {
    icon: [
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' }
    ],
  },
  manifest: '/site.webmanifest',
  keywords: `WHO fact sheets, health information retrieval, RAG AI, ChromaDB vector search, 
  global health data, disease facts, vaccine information, public health insights, medical research,
   WHO data`,
  authors: [{ name: 'MediFact Team', url: 'https://jimmyesang.vercel.app' }],
  creator: 'Jimmy Esang',
  publisher: 'Jimmy Esang',
  openGraph: {
    title: 'MediFact - WHO Fact Sheets AI Retrieval',
    description: `Access WHO health fact sheets instantly with semantic search. Accurate 
    global health information from authoritative sources.`,
    type: 'website',
    siteName: 'MediFact',
    images: [
      {
        url: '/cover.png',
        width: 1200,
        height: 630,
        alt: 'MediFact WHO Health Facts Retrieval'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'MediFact - WHO Fact Sheets',
    description: 'Instant access to WHO health data via AI-powered RAG search',
    images: ['/cover.png']
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1
    }
  },
  verification: {
    google: 'YUa7__6Sc2eo89jNPhxREJ1UkvWl4ZWXp1XvX2QW3r4',
  }
  }


export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>
          <QueryProvider>
            {children}
             <ToastContainer 
              position='bottom-right'
              newestOnTop={true}/>
          </QueryProvider>
        </Providers>
      </body>
    </html>
  );
}
