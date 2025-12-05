import "./globals.css";
import { ReactNode } from "react";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "Starter App",
  description: "Starter template with Next.js + Tailwind",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground font-sans flex flex-col">
        <Navbar />

        {/* Main content */}
        <main className="flex-1 container mx-auto px-4 mt-4 sm:px-6 lg:px-8">
          {children}
        </main>
      </body>
    </html>
  );
}
