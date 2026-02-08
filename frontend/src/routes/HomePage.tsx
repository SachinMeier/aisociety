import { useNavigate } from "react-router-dom";
import Divider from "../components/Divider";
import styles from "./HomePage.module.css";

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className={styles.container}>
      <div className={styles.crownIcon}>&#9813;</div>
      <h1 className={styles.title}>High Society</h1>
      <p className={styles.subtitle}>A game of wealth, status &amp; knowing when to fold</p>
      <Divider />
      <p className={styles.credit}>A Game by Reiner Knizia</p>
      <div className={styles.buttons}>
        <button className={styles.newGame} onClick={() => navigate("/new")}>
          New Game
        </button>
        <button className={styles.rulesButton} onClick={() => navigate("/rules")}>
          How to Play
        </button>
      </div>
    </div>
  );
}
