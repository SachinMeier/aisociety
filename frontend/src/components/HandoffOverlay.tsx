interface HandoffOverlayProps {
  playerName: string;
  onReveal: () => void;
}

export default function HandoffOverlay({
  playerName,
  onReveal,
}: HandoffOverlayProps) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(10, 20, 10, 0.97)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        fontFamily: "'Georgia', serif",
      }}
    >
      {/* Decorative flourish */}
      <div
        style={{
          fontSize: 28,
          color: "#d4af37",
          opacity: 0.3,
          marginBottom: 20,
          letterSpacing: 12,
        }}
      >
        &#x2767; &#x2619;
      </div>

      <div
        style={{
          fontSize: 14,
          color: "#8a7a60",
          textTransform: "uppercase",
          letterSpacing: 3,
          marginBottom: 8,
        }}
      >
        Pass the device to
      </div>

      <h1
        style={{
          color: "#ffd700",
          fontSize: 40,
          fontWeight: "bold",
          marginBottom: 36,
          textShadow: "0 2px 12px rgba(255,215,0,0.25)",
          letterSpacing: 1,
        }}
      >
        {playerName}
      </h1>

      <button
        onClick={onReveal}
        style={{
          backgroundColor: "transparent",
          color: "#d4af37",
          border: "2px solid rgba(212, 175, 55, 0.5)",
          borderRadius: 12,
          padding: "14px 36px",
          fontSize: 17,
          fontWeight: 600,
          cursor: "pointer",
          fontFamily: "'Georgia', serif",
          transition: "all 0.15s ease",
          letterSpacing: 0.5,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "rgba(212, 175, 55, 0.1)";
          e.currentTarget.style.borderColor = "#d4af37";
          e.currentTarget.style.transform = "translateY(-1px)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
          e.currentTarget.style.borderColor = "rgba(212, 175, 55, 0.5)";
          e.currentTarget.style.transform = "none";
        }}
      >
        I'm {playerName} — Reveal my hand
      </button>

      <p
        style={{
          color: "#5a5040",
          fontSize: 12,
          marginTop: 24,
          maxWidth: 280,
          textAlign: "center",
          lineHeight: 1.6,
          letterSpacing: 0.3,
        }}
      >
        Ensure no one else can see the screen before revealing your coins.
      </p>
    </div>
  );
}
