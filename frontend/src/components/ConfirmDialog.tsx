interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <div
      onClick={onCancel}
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        fontFamily: "'Georgia', serif",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: "rgba(20, 30, 20, 0.97)",
          border: "1.5px solid rgba(212, 175, 55, 0.4)",
          borderRadius: 14,
          padding: "24px 32px",
          maxWidth: 340,
          textAlign: "center",
          boxShadow: "0 8px 32px rgba(0,0,0,0.5), 0 0 20px rgba(212,175,55,0.08)",
        }}
      >
        <h2
          style={{
            color: "#f0e6d3",
            fontSize: 20,
            fontWeight: 600,
            marginBottom: 12,
            letterSpacing: 0.5,
          }}
        >
          {title}
        </h2>

        <p
          style={{
            color: "#a09080",
            fontSize: 15,
            lineHeight: 1.5,
            marginBottom: 24,
          }}
        >
          {message}
        </p>

        <div
          style={{
            display: "flex",
            gap: 12,
            justifyContent: "center",
          }}
        >
          <button
            onClick={onCancel}
            style={{
              padding: "10px 24px",
              borderRadius: 8,
              border: "2px solid rgba(255,255,255,0.2)",
              backgroundColor: "rgba(255,255,255,0.05)",
              color: "#d0d0c0",
              fontSize: 15,
              fontWeight: "bold",
              cursor: "pointer",
              fontFamily: "'Georgia', serif",
              transition: "all 0.1s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "#ffd700";
              e.currentTarget.style.color = "#ffd700";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.2)";
              e.currentTarget.style.color = "#d0d0c0";
            }}
          >
            {cancelLabel}
          </button>

          <button
            onClick={onConfirm}
            style={{
              padding: "10px 24px",
              borderRadius: 8,
              border: "none",
              backgroundColor: "#2d6b2d",
              color: "#fff",
              fontSize: 15,
              fontWeight: "bold",
              cursor: "pointer",
              fontFamily: "'Georgia', serif",
              boxShadow: "0 4px 12px rgba(45,107,45,0.4)",
              transition: "all 0.15s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "#3a8a3a";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "#2d6b2d";
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
