// components/Nav.tsx
//
// What it does: the navigation bar. Highlights the page you are on.
// Which files use it: app/layout.tsx.

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const PAGES = [
  { href: "/", label: "Race" },
  { href: "/head-to-head/", label: "Head to head" },
  { href: "/ranking/", label: "Ranking" },
  { href: "/drivers/", label: "Drivers" },
  { href: "/seasons/", label: "Seasons" },
  { href: "/about/", label: "About" },
];

export default function Nav() {
  const path = usePathname();
  const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href.slice(0, -1)));
  return (
    <nav className="nav">
      <Link href="/" className="brand">
        F1 <span>EQUALIZER</span>
      </Link>
      <div className="nav-links">
        {PAGES.map((page) => (
          <Link key={page.href} href={page.href} className={isActive(page.href) ? "active" : ""}>
            {page.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
