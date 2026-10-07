import { Link } from "react-router-dom";
import { MicroscopeIcon, ShieldIcon } from "./icons";

interface Props {
  variant?: "home" | "results";
}

export default function Header({ variant = "home" }: Props) {
  return (
    <header className={`header header--${variant}`}>
      <Link to="/" className="header__brand">
        <span className="header__logo">
          {variant === "home" ? <MicroscopeIcon /> : <ShieldIcon />}
        </span>
        <span>Verificador de Evidências</span>
      </Link>
    </header>
  );
}
