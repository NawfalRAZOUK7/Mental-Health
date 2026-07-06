import "./globals.css";

export const metadata = {
  title: "Mental Health Viz — Global Analytics & Prediction",
  description:
    "Global mental-health analytics on WHO & IHME data: leakage-free ML, data mining, deep-learning forecasting, and a live country-level suicide-rate predictor. Educational, non-clinical.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:wght@400;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
