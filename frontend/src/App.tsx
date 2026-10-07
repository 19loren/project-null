import { Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import ResultadosPage from "./pages/ResultadosPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/resultados" element={<ResultadosPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
