import React from "react";

export const metadata = {
  title: "BATS - Binary Options AI Trading Dashboard",
  description: "Automated binary options AI trading dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "sans-serif", backgroundColor: "#0f172a", color: "#f8fafc" }}>
        {children}
      </body>
    </html>
  );
}
