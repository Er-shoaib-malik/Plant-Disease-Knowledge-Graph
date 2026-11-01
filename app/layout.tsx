import './globals.css'
import '@xyflow/react/dist/style.css'
import type { Metadata } from 'next'
export const metadata: Metadata={title:'Plant Disease Intelligence',description:'Explainable plant disease prediction with sourced agricultural knowledge.'}
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
