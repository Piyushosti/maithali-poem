// "use client";
// import { useState, useEffect } from "react";
// import { Feather, Send, Loader2, History, Plus, MessageSquare, Copy, Trash2 } from "lucide-react";

// interface PoemHistory {
//   id: number;
//   word: string;
//   maithili: string;
//   poem: string;
// }

// export default function Home() {
//   const [word, setWord] = useState("");
//   const [result, setResult] = useState({ maithili: "", poem: "" });
//   const [loading, setLoading] = useState(false);
//   const [history, setHistory] = useState<PoemHistory[]>([]);
//   const [isSidebarOpen, setIsSidebarOpen] = useState(true);

//   // --- 1. PERSISTENCE LOGIC (LocalStorage) ---
  
//   // Load history from browser storage on initial startup
//   useEffect(() => {
//     const savedHistory = localStorage.getItem("maithili-poem-vault");
//     if (savedHistory) {
//       try {
//         setHistory(JSON.parse(savedHistory));
//       } catch (e) {
//         console.error("Failed to parse history", e);
//       }
//     }
//   }, []);

//   // Save to browser storage whenever history changes
//   useEffect(() => {
//     if (history.length > 0) {
//       localStorage.setItem("maithili-poem-vault", JSON.stringify(history));
//     }
//   }, [history]);

//   const clearHistory = () => {
//     if (confirm("Are you sure you want to delete all saved poems?")) {
//       setHistory([]);
//       localStorage.removeItem("maithili-poem-vault");
//     }
//   };

//   const generatePoem = async (selectedWord?: string) => {
//     const inputWord = selectedWord || word;
//     if (!inputWord) return;
    
//     setLoading(true);
//     try {
//       const response = await fetch("http://127.0.0.1:8000/generate", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ word: inputWord }),
//       });
//       const data = await response.json();
      
//       const newEntry = {
//         id: Date.now(),
//         word: inputWord,
//         maithili: data.maithili_input,
//         poem: data.poem,
//       };

//       setResult({ maithili: data.maithili_input, poem: data.poem });
//       setHistory((prev) => [newEntry, ...prev]);
//       setWord("");
//     } catch (error) {
//       console.error("Error calling backend:", error);
//       alert("Make sure your Python backend is running!");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="flex h-screen bg-[#fdf6e3] text-gray-900 font-serif">
//       {/* --- SIDEBAR (ChatGPT Style) --- */}
//       <aside className={`${isSidebarOpen ? "w-72" : "w-0"} transition-all duration-300 bg-[#171717] text-gray-200 flex flex-col overflow-hidden shadow-2xl`}>
//         <div className="p-4 flex gap-2">
//           <button 
//             onClick={() => {setResult({maithili: "", poem: ""}); setWord("");}}
//             className="flex-1 flex items-center justify-center gap-2 border border-white/20 rounded-lg py-2.5 hover:bg-white/10 transition text-sm font-sans"
//           >
//             <Plus size={16} /> New Poem
//           </button>
//         </div>
        
//         <div className="flex-1 overflow-y-auto px-3 custom-scrollbar">
//           <div className="flex justify-between items-center px-2 py-4">
//             <p className="text-[11px] font-bold uppercase tracking-widest text-gray-500 font-sans">Recent History</p>
//             {history.length > 0 && (
//               <button onClick={clearHistory} className="text-gray-500 hover:text-red-400 transition">
//                 <Trash2 size={12} />
//               </button>
//             )}
//           </div>
          
//           <div className="space-y-1">
//             {history.map((item) => (
//               <button
//                 key={item.id}
//                 onClick={() => setResult({ maithili: item.maithili, poem: item.poem })}
//                 className={`w-full flex items-center gap-3 p-3 rounded-lg text-left text-sm transition group hover:bg-[#2f2f2f] ${result.maithili === item.maithili ? 'bg-[#2f2f2f]' : ''}`}
//               >
//                 <MessageSquare size={14} className="flex-shrink-0 text-gray-500" />
//                 <span className="truncate font-sans">{item.word}</span>
//               </button>
//             ))}
//           </div>
//         </div>

//         <div className="p-4 border-t border-white/10 text-[10px] text-gray-500 text-center font-sans">
//           Saved Locally in Browser
//         </div>
//       </aside>

//       {/* --- MAIN WORKSPACE --- */}
//       <main className="flex-1 flex flex-col relative overflow-hidden bg-gradient-to-br from-[#fdf6e3] to-[#f5ecd5]">
//         {/* Toggle UI */}
//         <button 
//           onClick={() => setIsSidebarOpen(!isSidebarOpen)}
//           className="absolute top-6 left-6 p-2.5 bg-white/80 backdrop-blur rounded-xl shadow-sm border border-amber-200 z-10 hover:bg-white transition"
//         >
//           <History size={18} className="text-amber-900" />
//         </button>

//         <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-12 overflow-y-auto">
//           {!result.poem ? (
//             <div className="text-center space-y-6 max-w-md animate-in fade-in zoom-in duration-500">
//               <div className="bg-red-900 w-16 h-16 rounded-full flex items-center justify-center mx-auto shadow-xl">
//                 <Feather className="text-amber-100" size={32} />
//               </div>
//               <div>
//                 <h1 className="text-4xl font-bold text-red-900 mb-2">Maithili Nano-Poet</h1>
//                 <p className="text-amber-800/70 font-sans italic text-lg">"Where words find their rhythm in Mithila."</p>
//               </div>
//               <div className="grid grid-cols-2 gap-3 pt-4 font-sans text-sm">
//                 <button onClick={() => setWord("Mountain")} className="p-3 bg-white/50 border border-amber-200 rounded-lg hover:bg-white transition">"Mountain"</button>
//                 <button onClick={() => setWord("Peace")} className="p-3 bg-white/50 border border-amber-200 rounded-lg hover:bg-white transition">"Peace"</button>
//               </div>
//             </div>
//           ) : (
//             <div className="w-full max-w-2xl bg-white p-10 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.1)] border border-amber-100 animate-in fade-in slide-in-from-bottom-8 relative">
//                <button 
//                 onClick={() => {
//                   navigator.clipboard.writeText(result.poem);
//                   alert("Poem copied!");
//                 }}
//                 className="absolute top-6 right-6 p-2 text-gray-300 hover:text-red-800 hover:bg-red-50 rounded-full transition"
//               >
//                 <Copy size={20} />
//               </button>
              
//               <div className="mb-8">
//                 <span className="bg-red-50 text-red-700 px-3 py-1 rounded-full text-xs font-bold font-sans uppercase tracking-widest">
//                   Keyword: {result.maithili}
//                 </span>
//               </div>

//               <p className="text-2xl leading-[1.8] whitespace-pre-wrap italic text-gray-800 border-l-4 border-red-800 pl-8 drop-shadow-sm">
//                 {result.poem}
//               </p>
//             </div>
//           )}
//         </div>

//         {/* CHAT INPUT BAR */}
//         <div className="pb-10 px-6">
//           <div className="max-w-3xl mx-auto relative group">
//             <div className="absolute -inset-1 bg-gradient-to-r from-red-900 to-amber-700 rounded-2xl blur opacity-10 group-hover:opacity-20 transition duration-1000"></div>
//             <div className="relative flex gap-2 bg-white p-2.5 rounded-2xl shadow-2xl border border-amber-100">
//               <input
//                 type="text"
//                 className="flex-1 bg-transparent px-6 py-4 outline-none text-lg font-sans"
//                 placeholder="Write a poem about..."
//                 value={word}
//                 onChange={(e) => setWord(e.target.value)}
//                 onKeyDown={(e) => e.key === "Enter" && generatePoem()}
//               />
//               <button
//                 onClick={() => generatePoem()}
//                 disabled={loading}
//                 className="bg-red-900 text-white px-8 rounded-xl hover:bg-red-800 transition-all flex items-center justify-center disabled:bg-gray-300 shadow-lg shadow-red-900/20"
//               >
//                 {loading ? <Loader2 className="animate-spin" /> : <Send size={22} />}
//               </button>
//             </div>
//           </div>
//           <p className="text-center text-[10px] text-amber-900/40 mt-4 font-sans uppercase tracking-widest font-bold">
//             Powered by MaithiliNano Transformer Engine
//           </p>
//         </div>
//       </main>
//     </div>
//   );
// }


// 2nd version 

// "use client";
// import { useState, useEffect } from "react";
// import { Feather, Send, Loader2, History, Plus, MessageSquare, Copy, Trash2, Volume2, Info } from "lucide-react";

// interface PoemHistory {
//   id: number;
//   word: string;
//   maithili: string;
//   poem: string;
//   cultural_note: string;
//   audio_url: string;
// }

// export default function Home() {
//   const [word, setWord] = useState("");
//   const [result, setResult] = useState<Partial<PoemHistory>>({});
//   const [loading, setLoading] = useState(false);
//   const [history, setHistory] = useState<PoemHistory[]>([]);
//   const [isSidebarOpen, setIsSidebarOpen] = useState(true);

//   // Load History from LocalStorage
//   useEffect(() => {
//     const saved = localStorage.getItem("maithili-vault");
//     if (saved) setHistory(JSON.parse(saved));
//   }, []);

//   // Save History to LocalStorage
//   useEffect(() => {
//     localStorage.setItem("maithili-vault", JSON.stringify(history));
//   }, [history]);

//   const generatePoem = async () => {
//     if (!word) return;
//     setLoading(true);
//     try {
//       const response = await fetch("http://127.0.0.1:8000/generate", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ word }),
//       });
//       const data = await response.json();
      
//       const newEntry = {
//         id: Date.now(),
//         word,
//         maithili: data.maithili_input,
//         poem: data.poem,
//         cultural_note: data.cultural_note,
//         audio_url: data.audio_url
//       };

//       setResult(newEntry);
//       setHistory((prev) => [newEntry, ...prev]);
//       setWord("");
//     } catch (error) {
//       alert("Error: Is your backend running?");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="flex h-screen bg-[#F7F4ED] text-[#2D2A26]">
//       {/* --- SIDEBAR --- */}
//       <aside className={`${isSidebarOpen ? "w-72" : "w-0"} transition-all duration-300 bg-[#171717] text-gray-300 flex flex-col overflow-hidden border-r border-white/10 shadow-2xl`}>
//         <div className="p-4">
//           <button onClick={() => setResult({})} className="w-full flex items-center gap-2 border border-white/20 rounded-xl p-3 hover:bg-white/10 transition text-sm">
//             <Plus size={18} /> New Chat
//           </button>
//         </div>
        
//         <div className="flex-1 overflow-y-auto px-3 space-y-1">
//           <p className="px-3 py-4 text-[10px] font-bold uppercase tracking-widest text-gray-500">History</p>
//           {history.map((item) => (
//             <button
//               key={item.id}
//               onClick={() => setResult(item)}
//               className={`w-full flex items-center gap-3 p-3 rounded-xl text-left text-sm transition group ${result.id === item.id ? 'bg-[#2F2F2F] text-white' : 'hover:bg-[#2F2F2F]'}`}
//             >
//               <MessageSquare size={16} className="opacity-40" />
//               <span className="truncate">{item.word}</span>
//             </button>
//           ))}
//         </div>
//       </aside>

//       {/* --- MAIN PANEL --- */}
//       <main className="flex-1 flex flex-col relative overflow-hidden">
//         {/* Toggle Button */}
//         <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="absolute top-6 left-6 p-2 hover:bg-black/5 rounded-lg z-10 transition">
//           <History size={20} className="text-[#8B2622]" />
//         </button>

//         <div className="flex-1 flex flex-col items-center justify-center p-6 pt-20">
//           {result.poem ? (
//             <div className="w-full max-w-2xl animate-in fade-in slide-in-from-bottom-6 duration-700">
//               {/* The Poem Card */}
//               <div className="bg-white p-10 rounded-[2rem] shadow-[0_10px_40px_rgba(0,0,0,0.04)] border border-black/5 relative overflow-hidden">
//                 <div className="absolute top-0 left-0 w-1.5 h-full bg-[#8B2622]"></div>
                
//                 <div className="flex justify-between items-start mb-8">
//                   <span className="bg-[#8B2622]/10 text-[#8B2622] px-4 py-1 rounded-full text-xs font-bold uppercase tracking-tighter">
//                     Keyword: {result.maithili}
//                   </span>
//                   <div className="flex gap-2">
//                     <button onClick={() => new Audio(result.audio_url).play()} className="p-2 hover:bg-gray-100 rounded-full text-[#8B2622] transition"><Volume2 size={20} /></button>
//                     <button onClick={() => navigator.clipboard.writeText(result.poem || "")} className="p-2 hover:bg-gray-100 rounded-full text-gray-400 transition"><Copy size={20} /></button>
//                   </div>
//                 </div>

//                 <p className="text-2xl leading-[1.9] text-[#2D2A26] font-serif italic mb-8">
//                   {result.poem}
//                 </p>

//                 {/* Cultural Insight Card */}
//                 <div className="mt-8 p-5 bg-[#F7F4ED] rounded-2xl flex gap-4 items-start border border-black/5">
//                   <Info size={20} className="text-[#8B2622] flex-shrink-0 mt-0.5" />
//                   <p className="text-sm text-gray-600 leading-relaxed">
//                     <span className="font-bold text-[#8B2622]">Cultural Insight:</span> {result.cultural_note}
//                   </p>
//                 </div>
//               </div>
//             </div>
//           ) : (
//             <div className="text-center space-y-6 opacity-40">
//               <Feather size={60} className="mx-auto text-[#8B2622]" />
//               <h2 className="text-2xl font-serif">Maithili Nano Poet</h2>
//               <p className="text-sm">Type a word below to generate a cultural poem.</p>
//             </div>
//           )}
//         </div>

//         {/* --- INPUT AREA --- */}
//         <div className="max-w-3xl w-full mx-auto p-8 pt-0">
//           <div className="relative flex items-center bg-white rounded-[1.5rem] shadow-xl border border-black/5 p-2 pr-3">
//             <input
//               type="text"
//               className="flex-1 px-6 py-4 outline-none text-lg"
//               placeholder="A poem about beauty..."
//               value={word}
//               onChange={(e) => setWord(e.target.value)}
//               onKeyDown={(e) => e.key === "Enter" && generatePoem()}
//             />
//             <button
//               onClick={generatePoem}
//               disabled={loading}
//               className="bg-[#171717] text-white h-12 w-12 flex items-center justify-center rounded-xl hover:bg-[#2F2F2F] transition disabled:bg-gray-200"
//             >
//               {loading ? <Loader2 className="animate-spin text-gray-400" /> : <Send size={20} />}
//             </button>
//           </div>
//           <p className="text-[10px] text-center mt-4 text-gray-400 uppercase tracking-widest font-bold">
//             Powered by Deep-Translator & MaithiliNano
//           </p>
//         </div>
//       </main>
//     </div>
//   );
// }


"use client";
import { useState, useEffect, useRef } from "react";
import {
  Feather, Send, Loader2, History, Plus,
  MessageSquare, Copy, Volume2, Info, Check,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

interface PoemEntry {
  id: number;
  word: string;
  maithili: string;
  poem: string;
  cultural_note: string;
  audio_url: string;
}

// ─── small helpers ────────────────────────────────────────────────────────────
const STORAGE_KEY = "maithili-vault";

function loadHistory(): PoemEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function saveHistory(history: PoemEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

// ─── component ────────────────────────────────────────────────────────────────
export default function Home() {
  const [word, setWord]               = useState("");
  const [result, setResult]           = useState<PoemEntry | null>(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [history, setHistory]         = useState<PoemEntry[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [copied, setCopied]           = useState(false);
  const audioRef                      = useRef<HTMLAudioElement | null>(null);

  // Hydrate history from localStorage after mount (avoids SSR mismatch)
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const pushToHistory = (entry: PoemEntry) => {
    setHistory((prev) => {
      const next = [entry, ...prev].slice(0, 50); // cap at 50
      saveHistory(next);
      return next;
    });
  };

  const generatePoem = async () => {
    const trimmed = word.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);

    try {
      const res  = await fetch(`${API_BASE}/generate`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ word: trimmed }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Server error ${res.status}`);
      }

      const data = await res.json();

      if (data.error) throw new Error(data.error);

      const entry: PoemEntry = {
        id:           Date.now(),
        word:         trimmed,
        maithili:     data.maithili_input,
        poem:         data.poem,
        cultural_note: data.cultural_note ?? "",
        audio_url:    data.audio_url,
      };

      setResult(entry);
      pushToHistory(entry);
      setWord("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const playAudio = (url: string) => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    audioRef.current = new Audio(url);
    audioRef.current.play();
  };

  const copyPoem = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const startNew = () => {
    setResult(null);
    setError(null);
    setWord("");
  };

  return (
    <div className="flex h-screen bg-[#F7F4ED] text-[#2D2A26]">
      {/* ── SIDEBAR ─────────────────────────────────────────────────────── */}
      <aside
        className={`${
          sidebarOpen ? "w-72" : "w-0"
        } transition-all duration-300 bg-[#171717] text-gray-300 flex flex-col overflow-hidden border-r border-white/10 shadow-2xl`}
      >
        <div className="p-4">
          <button
            onClick={startNew}
            className="w-full flex items-center gap-2 border border-white/20 rounded-xl p-3 hover:bg-white/10 transition text-sm"
          >
            <Plus size={18} /> New Poem
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          <p className="px-3 py-4 text-[10px] font-bold uppercase tracking-widest text-gray-500">
            History
          </p>
          {history.length === 0 && (
            <p className="px-3 text-xs text-gray-600">No poems yet.</p>
          )}
          {history.map((item) => (
            <button
              key={item.id}
              onClick={() => setResult(item)}
              className={`w-full flex items-center gap-3 p-3 rounded-xl text-left text-sm transition ${
                result?.id === item.id
                  ? "bg-[#2F2F2F] text-white"
                  : "hover:bg-[#2F2F2F]"
              }`}
            >
              <MessageSquare size={16} className="opacity-40 flex-shrink-0" />
              <span className="truncate">{item.word}</span>
            </button>
          ))}
        </div>
      </aside>

      {/* ── MAIN PANEL ──────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebarOpen((v) => !v)}
          className="absolute top-6 left-6 p-2 hover:bg-black/5 rounded-lg z-10 transition"
          title="Toggle history"
        >
          <History size={20} className="text-[#8B2622]" />
        </button>

        {/* ── Content area ── */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 pt-20">
          {error && (
            <div className="w-full max-w-2xl mb-4 bg-red-50 border border-red-200 text-red-700 px-5 py-3 rounded-2xl text-sm">
              ⚠️ {error}
            </div>
          )}

          {result ? (
            <div className="w-full max-w-2xl animate-in fade-in slide-in-from-bottom-6 duration-700">
              <div className="bg-white p-10 rounded-[2rem] shadow-[0_10px_40px_rgba(0,0,0,0.04)] border border-black/5 relative overflow-hidden">
                {/* Red left accent bar */}
                <div className="absolute top-0 left-0 w-1.5 h-full bg-[#8B2622]" />

                <div className="flex justify-between items-start mb-8">
                  <span className="bg-[#8B2622]/10 text-[#8B2622] px-4 py-1 rounded-full text-xs font-bold uppercase tracking-tighter">
                    Keyword: {result.maithili}
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => playAudio(result.audio_url)}
                      className="p-2 hover:bg-gray-100 rounded-full text-[#8B2622] transition"
                      title="Listen"
                    >
                      <Volume2 size={20} />
                    </button>
                    <button
                      onClick={() => copyPoem(result.poem)}
                      className="p-2 hover:bg-gray-100 rounded-full text-gray-400 transition"
                      title="Copy poem"
                    >
                      {copied ? <Check size={20} className="text-green-500" /> : <Copy size={20} />}
                    </button>
                  </div>
                </div>

                {/* Poem */}
                <p className="text-2xl leading-[1.9] text-[#2D2A26] font-serif italic mb-8 whitespace-pre-line">
                  {result.poem}
                </p>

                {/* Cultural note */}
                {result.cultural_note && (
                  <div className="mt-8 p-5 bg-[#F7F4ED] rounded-2xl flex gap-4 items-start border border-black/5">
                    <Info size={20} className="text-[#8B2622] flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-gray-600 leading-relaxed">
                      <span className="font-bold text-[#8B2622]">Cultural Insight: </span>
                      {result.cultural_note}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            !loading && (
              <div className="text-center space-y-6 opacity-40">
                <Feather size={60} className="mx-auto text-[#8B2622]" />
                <h2 className="text-2xl font-serif">Maithili Nano Poet</h2>
                <p className="text-sm">Type any word below to generate a cultural poem.</p>
              </div>
            )
          )}

          {loading && (
            <div className="flex flex-col items-center gap-4 opacity-60">
              <Loader2 size={40} className="animate-spin text-[#8B2622]" />
              <p className="text-sm font-medium">Composing your poem…</p>
            </div>
          )}
        </div>

        {/* ── Input bar ── */}
        <div className="max-w-3xl w-full mx-auto p-8 pt-0">
          <div className="relative flex items-center bg-white rounded-[1.5rem] shadow-xl border border-black/5 p-2 pr-3">
            <input
              type="text"
              className="flex-1 px-6 py-4 outline-none text-lg bg-transparent"
              placeholder="e.g. home, river, moon…"
              value={word}
              onChange={(e) => setWord(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && generatePoem()}
              disabled={loading}
            />
            <button
              onClick={generatePoem}
              disabled={loading || !word.trim()}
              className="bg-[#171717] text-white h-12 w-12 flex items-center justify-center rounded-xl hover:bg-[#2F2F2F] transition disabled:bg-gray-200 disabled:cursor-not-allowed"
            >
              {loading
                ? <Loader2 size={20} className="animate-spin text-gray-400" />
                : <Send size={20} />}
            </button>
          </div>
          <p className="text-[10px] text-center mt-4 text-gray-400 uppercase tracking-widest font-bold">
            Powered by Deep-Translator &amp; MaithiliNano
          </p>
        </div>
      </main>
    </div>
  );
}