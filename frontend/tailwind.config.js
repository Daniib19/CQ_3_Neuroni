/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "var(--primary)",
        "primary-hover": "var(--primary-hover)",
        "primary-active": "var(--primary-active)",
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface1: "var(--surface-1)",
        surface2: "var(--surface-2)",
        border: "var(--border)",
        muted: "var(--muted)",
        success: "var(--success)",
        warning: "var(--warning)",
        error: "var(--error)",
      },
    },
  },
  plugins: [],
};
