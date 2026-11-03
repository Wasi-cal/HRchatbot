import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], weight: ["400", "500", "600", "700"], display: "swap" });

export const metadata: Metadata = {
  title: "Tessa · Company Policy Assistant",
  description: "Ask questions about company policies by chat or voice.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#4F46E5" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.className}>
      <body>{children}</body>
    </html>
  );
}
