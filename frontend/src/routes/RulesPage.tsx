import { useNavigate } from "react-router-dom";
import PaperCard from "../components/PaperCard";
import Coin from "../components/MoneyCard";
import StatusCard from "../components/StatusCard";
import Divider from "../components/Divider";
import type { StatusCardData } from "../types";
import styles from "./RulesPage.module.css";

const COIN_VALUES = [1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000];

const POSSESSIONS: StatusCardData[] = Array.from({ length: 10 }, (_, i) => ({
  kind: "possession",
  value: i + 1,
}));

const TITLES: StatusCardData[] = [
  { kind: "title" },
  { kind: "title" },
  { kind: "title" },
];

const MISFORTUNES: StatusCardData[] = [
  { kind: "misfortune", misfortune: "scandal" },
  { kind: "misfortune", misfortune: "debt" },
  { kind: "misfortune", misfortune: "theft" },
];

export default function RulesPage() {
  const navigate = useNavigate();

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.crownIcon}>&#9813;</div>
        <h1 className={styles.title}>High Society</h1>
        <p className={styles.tagline}>Money Isn't Everything</p>
        <p className={styles.credit}>A Game by Reiner Knizia</p>
        <Divider className={styles.headerDivider} />
      </header>

      <main className={styles.content}>
        <PaperCard className={styles.section}>
          <h2 className={styles.sectionTitle}>The Goal</h2>
          <p className={styles.text}>
            Acquire the most prestigious collection of Possession and Title cards while
            keeping more money than at least one opponent. The poorest player at game's
            end is eliminated—then the highest status wins.
          </p>
        </PaperCard>

        <PaperCard className={styles.section}>
          <h2 className={styles.sectionTitle}>The Cards</h2>

          <div className={styles.cardGroup}>
            <h3 className={styles.cardType}>Coins</h3>
            <p className={styles.text}>Each player starts with 11 coins:</p>
            <div className={styles.coinRow}>
              {COIN_VALUES.map((v) => (
                <Coin key={v} value={v} size="compact" />
              ))}
            </div>
          </div>

          <div className={styles.cardGroup}>
            <h3 className={styles.cardType}>Possessions <span className={styles.cardCount}>(10 cards)</span></h3>
            <p className={styles.text}>
              Worth 1–10 status points each. The foundation of your prestige.
            </p>
            <div className={styles.cardRow}>
              {POSSESSIONS.map((card) => (
                <StatusCard key={card.value} card={card} size="small" />
              ))}
            </div>
          </div>

          <div className={styles.cardGroup}>
            <h3 className={styles.cardType}>Titles <span className={styles.cardCount}>(3 cards)</span></h3>
            <p className={styles.text}>
              Each Title card <em>doubles</em> your total Possession value.
            </p>
            <div className={styles.cardRow}>
              {TITLES.map((card, i) => (
                <StatusCard key={i} card={card} size="small" />
              ))}
            </div>
          </div>

          <div className={styles.cardGroup}>
            <h3 className={styles.cardType}>Misfortunes <span className={styles.cardCount}>(3 cards)</span></h3>
            <p className={styles.text}>
              Misfortune cards hurt you. The first player to pass takes the card.
            </p>
            <div className={styles.cardRow}>
              {MISFORTUNES.map((card) => (
                <StatusCard key={card.misfortune} card={card} size="small" />
              ))}
            </div>
          </div>
        </PaperCard>

        <PaperCard className={styles.section}>
          <h2 className={styles.sectionTitle}>How to Play</h2>

          <div className={styles.ruleBlock}>
            <div className={styles.ruleCoin}>one</div>
            <div>
              <h4 className={styles.ruleTitle}>Reveal a Status Card</h4>
              <p className={styles.text}>A status card is flipped. Players will bid to win (or avoid) it.</p>
            </div>
          </div>

          <div className={styles.ruleBlock}>
            <div className={styles.ruleCoin}>two</div>
            <div>
              <h4 className={styles.ruleTitle}>Bid or Pass</h4>
              <p className={styles.text}>
                On your turn, either add coins to increase your bid, or pass. You cannot make change or remove coins from your bid.
                Passing returns your bid to your hand but removes you from the round. Once you Pass, you cannot reenter that round.
              </p>
            </div>
          </div>

          <div className={styles.ruleBlock}>
            <div className={styles.ruleCoin}>three</div>
            <div>
              <h4 className={styles.ruleTitle}>Win the Card</h4>
              <p className={styles.text}>
                <strong>Possessions & Titles:</strong> Last bidder standing wins the card. The coins they bid are discarded.<br />
                <strong>Misfortunes:</strong> First player to pass takes the card. Their bid returns to their hand; everyone else loses their bids.
              </p>
            </div>
          </div>
        </PaperCard>

        <PaperCard className={styles.section}>
          <h2 className={styles.sectionTitle}>Game End</h2>
          <p className={styles.text}>
            The game ends when the 4th multiplier card appears (3 Titles + Scandal).
            That card is <em>not</em> auctioned and scoring begins immediately.
          </p>
          <div className={styles.scoringBox}>
            <p><strong>Step 1:</strong> The player(s) with the least remaining money is eliminated.</p>
            <p><strong>Step 2:</strong> Remaining players sum their Possession values, apply Misfortunes, then apply any Titles.</p>
            <p><strong>Step 3:</strong> The player with the highest status wins. Ties are broken by remaining money and then the highest single possession value.</p>
          </div>
        </PaperCard>

        <PaperCard className={styles.section}>
          <h2 className={styles.sectionTitle}>Strategy Tips</h2>
          <ul className={styles.tipsList}>
            <li>Never be the poorest—the poorest always loses.</li>
            <li>In general, aim to be the second poorest player.</li>
            <li>Don't be afraid to pass. Patience wins auctions.</li>
            <li>Track opponents' spent money to gauge their remaining power.</li>
            <li>Reserve some money to defend against Misfortune cards.</li>
            <li>Keep track of how many red-bordered cards have been revealed. The game ends when the fourth one is revealed.</li>
          </ul>
        </PaperCard>
      </main>

      <footer className={styles.footer}>
        <button className={styles.backButton} onClick={() => navigate("/")}>
          Back to Home
        </button>
      </footer>
    </div>
  );
}
