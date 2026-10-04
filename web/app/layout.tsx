import "./globals.css";
export const metadata = { title: "MNIST CNN", description: "Draw a digit, let the CNN guess" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
