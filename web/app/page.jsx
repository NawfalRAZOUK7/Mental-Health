import SiteClient from "../components/SiteClient";
import predictions from "../data/predictions.json";

const scaleMax = Math.ceil(Math.max(...predictions.map((p) => p.upper)) / 5) * 5;

export default function Home() {
  return <SiteClient predictions={predictions} scaleMax={scaleMax} />;
}
