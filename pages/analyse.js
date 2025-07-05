import Link from "next/link";
import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

export default function Analyse() {
  const [chats, setChats] = useState([
    { id: 1, name: "Chat 1", messages: [] },
    { id: 2, name: "Chat 2", messages: [] },
    { id: 3, name: "Chat 3", messages: [] },
  ]);
  const [activeChat, setActiveChat] = useState(chats[0]);
  const [message, setMessage] = useState("");
  const [isAiTyping, setIsAiTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat.messages, isAiTyping]);

  const handleChatSelect = (chat) => {
    setActiveChat(chat);
  };

  const handleNewChat = () => {
    const newChat = {
      id: chats.length + 1,
      name: `Chat ${chats.length + 1}`,
      messages: [],
    };
    setChats([newChat, ...chats]);
    setActiveChat(newChat);
  };

  const sendMessage = (messageText) => {
    if (messageText.trim()) {
      const newMessage = {
        id: Date.now(),
        text: messageText,
        sender: 'user',
        timestamp: new Date(),
      };
      
      const updatedChat = {
        ...activeChat,
        messages: [...activeChat.messages, newMessage]
      };
      
      setChats(chats.map(chat => 
        chat.id === activeChat.id ? updatedChat : chat
      ));
      setActiveChat(updatedChat);
      setIsAiTyping(true);
      
      // Simulate AI typing and response
      setTimeout(() => {
        const aiResponse = {
          id: Date.now() + 1,
          text: getAIResponse(messageText),
          sender: 'ai',
          timestamp: new Date(),
        };
        
        const updatedChatWithAI = {
          ...updatedChat,
          messages: [...updatedChat.messages, aiResponse]
        };
        
        setChats(chats.map(chat => 
          chat.id === activeChat.id ? updatedChatWithAI : chat
        ));
        setActiveChat(updatedChatWithAI);
        setIsAiTyping(false);
      }, 1500 + Math.random() * 1000);
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

  const handleInputChange = (e) => {
    setMessage(e.target.value);
  };

  const formatTimestamp = (timestamp) => {
    const now = new Date();
    const messageTime = new Date(timestamp);
    const diffInMinutes = Math.floor((now - messageTime) / (1000 * 60));
    
    if (diffInMinutes < 1) {
      return 'Just now';
    } else if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`;
    } else if (diffInMinutes < 1440) {
      return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
      return messageTime.toLocaleDateString();
    }
  };

  const getUserAvatar = () => (
    <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center text-white text-sm font-medium shadow-md transform transition-all duration-200 hover:scale-110 hover:shadow-lg">
      U
    </div>
  );

  const getAIAvatar = () => (
    <div className="w-8 h-8 bg-gradient-to-br from-gray-600 to-gray-700 rounded-full flex items-center justify-center text-white text-sm font-medium shadow-md transform transition-all duration-200 hover:scale-110 hover:shadow-lg">
      <svg className="w-4 h-4 transform transition-transform duration-200 hover:rotate-12" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 1H5C3.89 1 3 1.89 3 3V7H2V9H3V15H2V17H3V21C3 22.1 3.89 23 5 23H19C20.1 23 21 22.1 21 21V17H22V15H21V9H22V7H21ZM19 9V15H5V9H19ZM9 11V13H7V11H9ZM13 11V13H11V11H13ZM17 11V13H15V11H17Z"/>
      </svg>
    </div>
  );

  const TypingIndicator = () => (
    <motion.div 
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex items-start space-x-3 mb-4"
    >
      {getAIAvatar()}
      <div className="bg-gray-200 rounded-2xl rounded-bl-md px-4 py-3 transition-all duration-300 shadow-sm hover:shadow-md">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </motion.div>
  );

  const MessageBubble = ({ message, isOwn }) => (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex items-start space-x-3 mb-4 ${isOwn ? 'flex-row-reverse space-x-reverse' : ''} group`}
    >
      {isOwn ? getUserAvatar() : getAIAvatar()}
      <div className="flex flex-col max-w-xs lg:max-w-md">
        <div className={`px-4 py-3 rounded-2xl transition-all duration-300 group-hover:shadow-lg group-hover:scale-[1.02] ${
          isOwn
            ? 'bg-gray-600 text-white rounded-br-md shadow-md hover:bg-gray-700'
            : 'bg-gray-200 text-black rounded-bl-md shadow-sm hover:bg-gray-300'
        }`}>
          <p className="text-sm leading-relaxed">{message.text}</p>
        </div>
        <div className={`flex items-center mt-1 ${isOwn ? 'justify-end' : 'justify-start'} opacity-0 group-hover:opacity-100 transition-opacity duration-200`}>
          <span className="text-xs text-gray-600">
            {formatTimestamp(message.timestamp)}
          </span>
        </div>
      </div>
    </motion.div>
  );

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.4,
        ease: "easeOut",
      },
    },
  };

  return (
    <div className="h-screen font-montserrat bg-white text-gray-900 transition-colors duration-300 overflow-hidden flex flex-col">
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="flex justify-center pt-8 pb-4 flex-shrink-0"
      >
        <div className="flex bg-gray-200 rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full bg-gray-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              View
            </div>
          </Link>
        </div>
      </motion.nav>

      {/* Page Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
        className="px-8 py-4 flex-shrink-0"
      >
        <h1 className="text-3xl font-bold text-center text-black transform transition-all duration-300">Analyse Your Data</h1>
        <p className="text-center text-gray-600 mt-2">Chat with our AI to analyze and understand your data better</p>
      </motion.div>

      {/* Main Layout */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.2 }}
        className="flex flex-1 min-h-0"
      >
        {/* Left Sidebar */}
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="w-64 bg-white flex flex-col transition-all duration-300 shadow-lg"
        >
          {/* Chat History */}
          <div className="flex-1 p-4 min-h-0 overflow-hidden">
            <h3 className="font-medium text-sm mb-4 text-black">Chat History</h3>
            <div className="space-y-2">
              {chats.map((chat, index) => (
                <motion.div
                  key={chat.id}
                  variants={itemVariants}
                  onClick={() => handleChatSelect(chat)}
                  className={`p-3 rounded-lg cursor-pointer transition-all duration-200 transform hover:scale-[1.01] hover:shadow-md active:scale-95 ${
                    activeChat.id === chat.id
                      ? 'bg-gray-600 text-white shadow-lg scale-[1.01]'
                      : 'hover:bg-gray-300 text-black'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-medium truncate flex-1 mr-2">{chat.name}</span>
                    {chat.messages.length > 0 && (
                      <span className={`text-xs opacity-75 px-2 py-1 rounded-full transition-all duration-200 flex-shrink-0 ${
                        activeChat.id === chat.id ? 'bg-white/20 hover:bg-white/30' : 'bg-gray-400 hover:bg-gray-500'
                      }`}>
                        {chat.messages.length}
                      </span>
                    )}
                  </div>
                  {chat.messages.length > 0 && (
                    <p className="text-xs opacity-75 mt-1 truncate">
                      {chat.messages[chat.messages.length - 1].text}
                    </p>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
          
          {/* New Chat Button */}
          <motion.div 
            variants={itemVariants}
            className="p-4 flex-shrink-0"
          >
            <button
              onClick={handleNewChat}
              className="w-full py-3 rounded-lg bg-gray-600 text-white font-medium hover:bg-gray-700 transition-all duration-200 flex items-center justify-center gap-2 transform hover:scale-[1.01] active:scale-95 shadow-md hover:shadow-lg"
            >
              <svg className="w-4 h-4 transform transition-transform duration-200 group-hover:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Chat
            </button>
          </motion.div>
        </motion.div>

        {/* Main Content */}
        <motion.main 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, ease: "easeOut", delay: 0.3 }}
          className="flex-1 flex flex-col bg-white min-h-0"
        >
          {/* Chat Name Header */}
          <div className="p-4 bg-white transition-all duration-300 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-black">{activeChat.name}</h2>
              </div>
            </div>
          </div>
          
          {/* Chat Area */}
          <div className="flex-1 p-8 overflow-y-auto min-h-0">
            <div className="max-w-4xl mx-auto">
              {activeChat.messages.length === 0 ? (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="text-center py-12"
                >

                  <h3 className="text-lg font-medium mb-2 text-black mt-4">Start Your Analysis</h3>
                  <p className="text-gray-600 mb-6">
                    Ask me anything about your data and I'll help you discover insights.
                  </p>
                  <div className="flex flex-wrap gap-3 justify-center">
                    <button 
                      onClick={() => handleSuggestionClick("Show me a summary")}
                      className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                    >
                      Show me a summary
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick("Find trends")}
                      className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                    >
                      Find trends
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick("Create visualization")}
                      className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                    >
                      Create visualization
                    </button>
                  </div>
                </motion.div>
              ) : (
                <div>
                  {activeChat.messages.map((msg) => (
                    <MessageBubble 
                      key={msg.id} 
                      message={msg} 
                      isOwn={msg.sender === 'user'} 
                    />
                  ))}
                  {isAiTyping && <TypingIndicator />}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          </div>
          
          {/* Input Area */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut", delay: 0.4 }}
            className="p-8 pt-0 flex-shrink-0"
          >
            <div className="relative max-w-4xl mx-auto">
              <div className="relative bg-gray-200 rounded-2xl transition-all duration-300 focus-within:border-gray-500 focus-within:shadow-xl focus-within:scale-[1.01] shadow-md">
                <textarea
                  value={message}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full h-12 p-4 pr-12 bg-transparent resize-none focus:outline-none placeholder-gray-600 text-black transition-all duration-200 overflow-hidden"
                  placeholder="Ask your AI data analyst a question about your data"
                  rows="1"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-2">
                  <button 
                    onClick={handleSendMessage}
                    disabled={!message.trim()}
                    className="w-8 h-8 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center transform hover:scale-110 active:scale-90 disabled:hover:scale-100 shadow-md hover:shadow-lg"
                  >
                    <svg className="w-4 h-4Removed  transform transition-transform duration-200 hover:translate-x-0.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
                    </svg>
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-600 mt-2 text-center transition-opacity duration-200">
                Press Enter to send, Shift+Enter for new line
              </p>
            </div>
          </motion.div>
        </motion.main>
      </motion.div>
    </div>
  );
}