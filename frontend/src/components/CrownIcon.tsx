const style: React.CSSProperties = {
  fontSize: "3.5rem",
  color: "#d4af37",
  filter: "drop-shadow(0 2px 8px rgba(212, 175, 55, 0.5))",
  lineHeight: 1,
  animation: "crownEntrance 0.6s ease-out",
};

export default function CrownIcon() {
  return (
    <>
      <style>{`
        @keyframes crownEntrance {
          from { opacity: 0; transform: translateY(-20px) scale(0.6); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
      <div style={style}>&#9813;</div>
    </>
  );
}
