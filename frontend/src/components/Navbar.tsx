"use client";

import Link from "next/link";
import { useState } from "react";

import { Bars3Icon, XMarkIcon } from "@heroicons/react/24/outline";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const routes = [
    { linkURL: "/",        label: "Home" },
    { linkURL: "/about",   label: "About" },
    { linkURL: "/contact", label: "Contact" },
  ];

  return (
    <nav className="bg-background text-primary shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          {/* Brand */}
          <div className="shrink-0">
            <Link href="/">
              <span className="text-2xl font-bold">Starter</span>
            </Link>
          </div>

          {/* Desktop menu */}
          <div className="hidden md:flex gap-2">
            {routes.map((route) => (
              <Link
                key={route.label}
                href={route.linkURL}
                className="hover:bg-(--surface-2) transition px-3 py-1 rounded-md"
              >
                {route.label}
              </Link>
            ))}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-foreground focus:outline-none"
            >
              {isOpen ? (
                <XMarkIcon className="size-6 cursor-pointer" />
              ) : (
                <Bars3Icon className="size-6 cursor-pointer" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden absolute top-16 left-0 w-full bg-[var(--background)] shadow-md z-50 pb-2">
          {routes.map((route) => (
            <Link
              key={route.label}
              href={route.linkURL}
              onClick={() => setIsOpen(false)}
              className="block px-3 py-2 rounded hover:bg-(--surface-2)"
            >
              {route.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
