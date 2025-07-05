import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { useState, useEffect } from "react";

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
  const [darkMode, setDarkMode] = useState(false);
  const [message, setMessage] = useState("");

  // Dark mode persistence
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedDarkMode !== null) {
      setDarkMode(JSON.parse(savedDarkMode));
    } else {
      setDarkMode(prefersDark);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

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

  const sendMessage = (messageText) => {
    if (messageText.trim()) {
      const newMessage = {
        id: Date.now(),
        text: messageText,
        sender: 'user',
        timestamp: new Date().toLocaleTimeString(),
      };
      
      const updatedChat = {
        ...activeChat,
        messages: [...activeChat.messages, newMessage]
      };
      
      setChats(chats.map(chat => 
        chat.id === activeChat.id ? updatedChat : chat
      ));
      setActiveChat(updatedChat);
      
      // Simulate AI response after a short delay
      setTimeout(() => {
        const aiResponse = {
          id: Date.now() + 1,
          text: getAIResponse(messageText),
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString(),
        };
        
        const updatedChatWithAI = {
          ...updatedChat,
          messages: [...updatedChat.messages, aiResponse]
        };
        
        setChats(chats.map(chat => 
          chat.id === activeChat.id ? updatedChatWithAI : chat
        ));
        setActiveChat(updatedChatWithAI);
      }, 1000);
    }
  };

  const getAIResponse = (userMessage) => {
    const lowerMessage = userMessage.toLowerCase();
    
    if (lowerMessage.includes('summary')) {
      return "I'd be happy to provide a summary of your data! Please upload your dataset first, and I'll analyze the key metrics, data distribution, and provide insights about your data structure.";
    } else if (lowerMessage.includes('trends')) {
      return "I can help you identify trends in your data! Once you upload your dataset, I'll look for patterns, correlations, and time-based trends that might be valuable for your analysis.";
    } else if (lowerMessage.includes('visualization') || lowerMessage.includes('chart')) {
      return "I can create various visualizations for your data! I can generate charts, graphs, and plots to help you better understand your data. Please upload your dataset and let me know what type of visualization you'd like.";
    } else {
      return "I'm here to help you analyze your data! You can ask me to summarize your data, find trends, create visualizations, or perform specific analyses. Please upload your dataset first so I can assist you better.";
    }
  };

  const handleSendMessage = () => {
    sendMessage(message);
    setMessage("");
  };

  const handleSuggestionClick = (suggestion) => {
    sendMessage(suggestion);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)] bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300`}>
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-full p-1 transition-colors duration-300">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 font-medium text-sm transition-all duration-300 cursor-pointer">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer">
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
        <div className="w-64 bg-gray-100 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col transition-colors duration-300">
          {/* Chat History */}
          <div className="flex-1 p-4">
            <h3 className="font-medium text-sm mb-4 text-gray-700 dark:text-gray-300">Chat History</h3>
            <div className="space-y-2">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => handleChatSelect(chat)}
                  className={`p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                    activeChat.id === chat.id
                      ? 'bg-blue-600 dark:bg-blue-500 text-white shadow-md'
                      : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{chat.name}</span>
                    {chat.messages.length > 0 && (
                      <span className="text-xs opacity-75">
                        {chat.messages.length} messages
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* New Chat Button */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={handleNewChat}
              className="w-full py-3 rounded-lg bg-blue-600 dark:bg-blue-500 text-white font-medium hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors duration-200 flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Chat
            </button>
          </div>
        </div>

        {/* Main Content */}
        <main className="flex-1 flex flex-col bg-white dark:bg-gray-900">
          {/* Chat Name Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 transition-colors duration-300">
            <h2 className="text-xl font-semibold">{activeChat.name}</h2>
          </div>
          
          {/* Chat Area */}
          <div className="flex-1 p-8 overflow-y-auto">
            <div className="space-y-4 max-w-4xl mx-auto">
              {activeChat.messages.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                    <svg className="w-8 h-8 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium mb-2">Start Your Analysis</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Ask me anything about your data and I'll help you discover insights.
                  </p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <button 
                      onClick={() => handleSuggestionClick("Show me a summary")}
                      className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors duration-200 text-sm"
                    >
                      Show me a summary
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick("Find trends")}
                      className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors duration-200 text-sm"
                    >
                      Find trends
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick("Create visualization")}
                      className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors duration-200 text-sm"
                    >
                      Create visualization
                    </button>
                  </div>
                </div>
              ) : (
                activeChat.messages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                      msg.sender === 'user'
                        ? 'bg-blue-600 text-white rounded-br-md'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-bl-md'
                    } transition-colors duration-300`}>
                      <p className="text-sm">{msg.text}</p>
                      <p className="text-xs mt-1 opacity-75">{msg.timestamp}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
          
          {/* Input Area */}
          <div className="p-8 pt-0">
            <div className="relative max-w-4xl mx-auto">
              <div className="relative bg-gray-100 dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 transition-colors duration-300 focus-within:border-blue-500 dark:focus-within:border-blue-400">
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="w-full min-h-12 max-h-32 p-4 pr-12 bg-transparent resize-none focus:outline-none placeholder-gray-500 dark:placeholder-gray-400"
                  placeholder="Ask your AI data analyst a question about your data"
                  rows="1"
                />
                <button 
                  onClick={handleSendMessage}
                  disabled={!message.trim()}
                  className="absolute right-2 top-2 w-8 h-8 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
                  </svg>
                </button>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
                Press Enter to send, Shift+Enter for new line
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}