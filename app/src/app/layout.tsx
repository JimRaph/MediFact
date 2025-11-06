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
  title: 'Health Info Hub - WHO Fact Sheets',
  description: 'Get accurate health information from WHO fact sheets',
  
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
