"use client";

import React, { useEffect, useState } from "react";

interface HealthStatus {
  status: string;
  database: string;
  redis: string;
}

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch((err) => console.error("Health check error:", err));
  }, []);

  return (
    <main style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      <header style={{ marginBottom: "2rem", borderBottom: "1px solid #334155", paddingBottom: "1rem" }}>
        <h1 style={{ color: "#38bdf8", fontSize: "2rem" }}>BATS — Binary Options AI Trading System</h1>
        <p style={{ color: "#94a3b8" }}>Phase 0 Project Foundation Dashboard</p>
      </header>

      <section style={{ backgroundColor: "#1e293b", padding: "1.5rem", borderRadius: "0.5rem", marginBottom: "2rem" }}>
        <h2>System Status</h2>
        {health ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginTop: "1rem" }}>
            <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.25rem", borderLeft: "4px solid #22c55e" }}>
              <span style={{ fontSize: "0.875rem", color: "#94a3b8" }}>Backend</span>
              <p style={{ fontSize: "1.25rem", fontWeight: "bold", margin: "0.5rem 0 0" }}>{health.status}</p>
            </div>
            <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.25rem", borderLeft: `4px solid ${health.database === 'connected' ? '#22c55e' : '#ef4444'}` }}>
              <span style={{ fontSize: "0.875rem", color: "#94a3b8" }}>Database</span>
              <p style={{ fontSize: "1.25rem", fontWeight: "bold", margin: "0.5rem 0 0" }}>{health.database}</p>
            </div>
            <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.25rem", borderLeft: `4px solid ${health.redis === 'connected' ? '#22c55e' : '#ef4444'}` }}>
              <span style={{ fontSize: "0.875rem", color: "#94a3b8" }}>Redis Cache</span>
              <p style={{ fontSize: "1.25rem", fontWeight: "bold", margin: "0.5rem 0 0" }}>{health.redis}</p>
            </div>
          </div>
        ) : (
          <p style={{ color: "#94a3b8" }}>Connecting to backend services...</p>
        )}
      </section>
    </main>
  );
}
