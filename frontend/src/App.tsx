import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./routes/HomePage";
import NewGamePage from "./routes/NewGamePage";
import GameTablePage from "./routes/GameTablePage";
import ResultPage from "./routes/ResultPage";
import RulesPage from "./routes/RulesPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/rules" element={<RulesPage />} />
        <Route path="/new" element={<NewGamePage />} />
        <Route path="/game/:id" element={<GameTablePage />} />
        <Route path="/game/:id/result" element={<ResultPage />} />
      </Routes>
    </BrowserRouter>
  );
}
