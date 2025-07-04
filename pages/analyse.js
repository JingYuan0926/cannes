import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { useState } from "react";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function Analyse() {
  const [chats, setChats] = useState([
    { id: 1, name: "Chat 1", messages: [] },
    { id: 2, name: "Chat 2", messages: [] },
    { id: 3, name: "Chat 3", messages: [] },
  ]);
  const [activeChat, setActiveChat] = useState(chats[0]);

  const handleChatSelect = (chat) => {
    setActiveChat(chat);
  };

  const handleNewChat = () => {
    const newChat = {
      id: chats.length + 1,
      name: `Chat ${chats.length + 1}`,
      messages: [],
    };
    setChats([...chats, newChat]);
    setActiveChat(newChat);
  };
  return (
    <div
      className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)]`}
    >
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
        <div className="flex bg-black/[.05] dark:bg-white/[.06] rounded-full p-1">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full bg-foreground text-background font-medium text-sm transition-colors cursor-pointer">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-black/[.05] dark:hover:bg-white/[.06] font-medium text-sm transition-colors cursor-pointer">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-black/[.05] dark:hover:bg-white/[.06] font-medium text-sm transition-colors cursor-pointer">
              View
            </div>
          </Link>
        </div>
      </nav>

      {/* Page Header */}
      <div className="px-8 py-4">
        <h1 className="text-3xl font-bold text-center">Analyse Your Data</h1>
      </div>

      {/* Main Layout */}
      <div className="flex min-h-[calc(100vh-180px)]">
        {/* Left Sidebar */}
        <div className="w-64 bg-black/[.05] dark:bg-white/[.06] border-r border-gray-200 dark:border-gray-700 flex flex-col">
          {/* Chat History */}
          <div className="flex-1 p-4">
            <h3 className="font-medium text-sm mb-4">Chat History</h3>
            <div className="space-y-2">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => handleChatSelect(chat)}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    activeChat.id === chat.id
                      ? 'bg-foreground text-background'
                      : 'hover:bg-black/[.05] dark:hover:bg-white/[.06]'
                  }`}
                >
                  {chat.name}
                </div>
              ))}
            </div>
          </div>
          
          {/* New Chat Button */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={handleNewChat}
              className="w-full py-3 rounded-lg bg-foreground text-background font-medium hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors"
            >
              New Chat
            </button>
          </div>
        </div>

        {/* Main Content */}
        <main className="flex-1 flex flex-col">
          {/* Chat Name Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold">{activeChat.name}</h2>
          </div>
          
          {/* Chat Area */}
          <div className="flex-1 p-8">
            <div className="space-y-4">
              {/* Chat messages will appear here */}
            </div>
          </div>
          
          {/* Input Area */}
          <div className="p-8 pt-0">
            <div className="relative max-w-4xl mx-auto">
              <textarea
                className="w-full h-12 p-3 pr-12 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 overflow-hidden"
                placeholder="Ask your AI data analyst a question about your data"
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    // Handle send message
                  }
                }}
              />
              <button className="absolute right-2 top-2 w-8 h-8 bg-foreground text-background rounded-md hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors flex items-center justify-center">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
                </svg>
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}