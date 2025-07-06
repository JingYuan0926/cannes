import Link from "next/link";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeRaw from 'rehype-raw';
import { readContentWithType } from "../utils/readFromWalrus";
import WalletConnect from '../components/WalletConnect';

export default function Analyse() {
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [message, setMessage] = useState("");
  const [isAiTyping, setIsAiTyping] = useState(false);
  const [activeFiles, setActiveFiles] = useState([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages, isAiTyping]);

  // Load active files from localStorage
  useEffect(() => {
    const loadActiveFiles = () => {
      try {
        const storedFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
        console.log('All stored files:', storedFiles);
        const activeFiles = storedFiles.filter(file => file.isActive);
        console.log('Active files found:', activeFiles);
        setActiveFiles(activeFiles);
      } catch (error) {
        console.error('Failed to load active files:', error);
        setActiveFiles([]);
      }
    };
    
    loadActiveFiles();
    
    // Listen for storage changes
    const handleStorageChange = () => {
      loadActiveFiles();
    };
    
    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('walrusFileUploaded', handleStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('walrusFileUploaded', handleStorageChange);
    };
  }, []);

  const handleChatSelect = (chat) => {
    setActiveChat(chat);
  };

  const handleNewChat = () => {
    // Don't create chat immediately - just clear active chat
    // Chat will be created when user sends first message
    setActiveChat(null);
  };

  const toggleSidebar = () => {
    setSidebarVisible(!sidebarVisible);
  };

  const loadActiveFilesContent = async () => {
    if (activeFiles.length === 0) {
      console.log('No active files to load');
      return '';
    }

    console.log(`Loading content for ${activeFiles.length} active files:`, activeFiles);
    setIsLoadingFiles(true);
    let combinedContent = '';
    
    try {
      for (const file of activeFiles) {
        try {
          console.log(`Loading content for file: ${file.name} (blobId: ${file.blobId})`);
          
          // Try to extract text using the new API endpoint
          const extractResponse = await fetch('/api/walrus/extract-text', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              blobId: file.blobId,
              fileName: file.name,
              fileType: file.type
            })
          });
          
          if (extractResponse.ok) {
            const extractResult = await extractResponse.json();
            console.log(`Text extracted for ${file.name}, length: ${extractResult.length}`);
            combinedContent += `\n\n--- File: ${file.name} ---\n${extractResult.text}`;
          } else {
            // Fallback to reading as raw content
            console.log(`Text extraction failed for ${file.name}, trying raw content`);
            const result = await readContentWithType(file.blobId);
            console.log(`File ${file.name} result:`, {
              isText: result.isText,
              contentType: result.contentType,
              contentLength: result.content?.length || 0
            });
            
            if (result.isText && result.content) {
              combinedContent += `\n\n--- File: ${file.name} ---\n${result.content}`;
              console.log(`Added text content for ${file.name}, length: ${result.content.length}`);
            } else {
              combinedContent += `\n\n--- File: ${file.name} ---\n[Binary file - cannot read content]`;
              console.log(`Skipped binary file: ${file.name}`);
            }
          }
        } catch (error) {
          console.error(`Failed to load file ${file.name}:`, error);
          combinedContent += `\n\n--- File: ${file.name} ---\n[Error loading file: ${error.message}]`;
        }
      }
    } catch (error) {
      console.error('Error loading active files:', error);
    } finally {
      setIsLoadingFiles(false);
    }
    
    console.log('Final combined content length:', combinedContent.length);
    console.log('Combined content preview:', combinedContent.substring(0, 500) + '...');
    return combinedContent;
  };

  const sendMessage = async (messageText) => {
    if (messageText.trim()) {
      const newMessage = {
        id: Date.now(),
        text: messageText,
        sender: 'user',
        timestamp: new Date(),
      };
      
      const isNewChat = !activeChat;
      let currentMessages = activeChat ? activeChat.messages : [];
      
      // Create temporary state for displaying the user message (don't add to chat list yet)
      const tempMessages = [...currentMessages, newMessage];
      
      if (activeChat) {
        // Update existing chat
        const updatedChat = {
          ...activeChat,
          messages: tempMessages
        };
        setChats(chats.map(chat => 
          chat.id === activeChat.id ? updatedChat : chat
        ));
        setActiveChat(updatedChat);
      } else {
        // For new conversation, create temporary display state (not in sidebar yet)
        setActiveChat({
          id: Date.now(),
          name: "",
          messages: tempMessages
        });
      }
      
      setIsAiTyping(true);
      
      try {
        // Load active files content
        const filesContent = await loadActiveFilesContent();
        console.log('About to send to API:', {
          promptLength: messageText.length,
          filesContentLength: filesContent.length,
          sampleDataReady: filesContent.length > 0,
          conversationLength: tempMessages.length,
          isNewChat: isNewChat
        });
        
        // Prepare conversation for API
        const conversation = tempMessages;
        
        const res = await fetch('/api/chatbot', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            prompt: messageText,
            sampleDataReady: filesContent.length > 0,
            conversation: conversation,
            filesContent: filesContent,
            generateChatName: isNewChat // Request chat name generation for new chats
          }),
        });
        
        if (!res.ok) throw new Error(`Error: ${res.status}`);
        const result = await res.json();
        
        const aiResponse = {
          id: Date.now() + 1,
          text: result.message,
          sender: 'ai',
          timestamp: new Date(),
        };
        
        const finalMessages = [...tempMessages, aiResponse];
        
        if (isNewChat) {
          // Create the chat ONLY after AI responds with name
          const newChat = {
            id: Date.now(),
            name: result.chatName || `Chat about ${messageText.substring(0, 30)}...`,
            messages: finalMessages
          };
          setChats([newChat, ...chats]);
          setActiveChat(newChat);
        } else {
          // Update existing chat
          const updatedChat = {
            ...activeChat,
            messages: finalMessages
          };
          setChats(chats.map(chat => 
            chat.id === activeChat.id ? updatedChat : chat
          ));
          setActiveChat(updatedChat);
        }
        
      } catch (error) {
        console.error('Error sending message:', error);
        const errorResponse = {
          id: Date.now() + 1,
          text: 'Sorry, I encountered an error. Please try again.',
          sender: 'ai',
          timestamp: new Date(),
        };
        
        const finalMessages = [...tempMessages, errorResponse];
        
        if (isNewChat) {
          // Create chat even on error for new chats
          const newChat = {
            id: Date.now(),
            name: `Error - ${messageText.substring(0, 30)}...`,
            messages: finalMessages
          };
          setChats([newChat, ...chats]);
          setActiveChat(newChat);
        } else {
          // Update existing chat
          const updatedChat = {
            ...activeChat,
            messages: finalMessages
          };
          setChats(chats.map(chat => 
            chat.id === activeChat.id ? updatedChat : chat
          ));
          setActiveChat(updatedChat);
        }
      } finally {
        setIsAiTyping(false);
      }
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
          {message.sender === 'ai' && !isOwn ? (
            <div className="react-markdown text-sm leading-relaxed">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkBreaks]}
                rehypePlugins={[rehypeRaw]}
                skipHtml={false}
              >
                {message.text}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="text-sm leading-relaxed">{message.text}</p>
          )}
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
      <style jsx>{`
        .react-markdown a {
          color: #4fc3f7 !important;
          text-decoration: underline;
          cursor: pointer;
        }
        .react-markdown a:hover {
          color: #81d4fa !important;
        }
      `}</style>
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative flex justify-center pt-8 pb-4 px-8 flex-shrink-0"
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
        <div className="absolute right-8 top-8">
          <WalletConnect />
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
        {/* Full Height Chat History Sidebar */}
        <AnimatePresence>
          {sidebarVisible && (
            <motion.div 
              variants={containerVariants}
              initial={{ opacity: 0, x: -264 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -264 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="w-64 bg-white flex flex-col transition-all duration-300 shadow-lg h-full"
            >
              {/* Chat History */}
              <div className="flex-1 min-h-0 flex flex-col">
                <h3 className="font-medium text-sm p-4 pb-2 text-black">Chat History</h3>
                <div className="flex-1 px-4 pb-4 overflow-y-auto">
                  <div className="space-y-2">
                    {chats.length === 0 ? (
                      <div className="text-center py-8">
                        <p className="text-sm text-gray-500 mb-4">No chats yet</p>
                        <p className="text-xs text-gray-400">Start a conversation to create your first chat</p>
                      </div>
                    ) : (
                      chats.map((chat, index) => (
                        <motion.div
                          key={chat.id}
                          variants={itemVariants}
                          onClick={() => handleChatSelect(chat)}
                          className={`p-3 rounded-lg cursor-pointer transition-all duration-200 transform hover:scale-[1.01] hover:shadow-md active:scale-95 ${
                            activeChat?.id === chat.id
                              ? 'bg-gray-600 text-white shadow-lg scale-[1.01]'
                              : 'hover:bg-gray-300 text-black'
                          }`}
                        >
                          <div className="flex justify-between items-center">
                            <span className="font-medium truncate flex-1 mr-2">{chat.name}</span>
                            {chat.messages.length > 0 && (
                              <span className={`text-xs opacity-75 px-2 py-1 rounded-full transition-all duration-200 flex-shrink-0 ${
                                activeChat?.id === chat.id ? 'bg-white/20 hover:bg-white/30' : 'bg-gray-400 hover:bg-gray-500'
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
                      ))
                    )}
                  </div>
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
          )}
        </AnimatePresence>

        {/* Main Content */}
        <motion.main 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, ease: "easeOut", delay: 0.3 }}
          className="flex-1 flex flex-col bg-white min-h-0"
        >
          {/* Sidebar Toggle Button */}
          <div className="p-4 flex justify-start">
            <button
              onClick={toggleSidebar}
              className="p-3 rounded-2xl bg-gray-200 hover:bg-gray-300 transition-all duration-300 ease-in-out shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95 flex items-center justify-center"
              title={sidebarVisible ? "Hide Chat History" : "Show Chat History"}
            >
              {sidebarVisible ? (
                <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
            </button>
          </div>
          
          {/* Chat Area */}
          <div className="flex-1 p-8 overflow-y-auto min-h-0">
            <div className="max-w-4xl mx-auto">
              {!activeChat || activeChat.messages.length === 0 ? (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="text-center py-12"
                >
                  <h3 className="text-lg font-medium mb-2 text-black mt-4">Start Your Analysis</h3>
                  <p className="text-gray-600 mb-4">
                    Ask me anything about your data and I'll help you discover insights.
                  </p>
                  <div className="mb-6 p-4 bg-gray-100 rounded-lg">
                    <p className="text-sm text-gray-700">
                      <strong>Active Files:</strong> {activeFiles.length} file{activeFiles.length !== 1 ? 's' : ''} ready for analysis
                    </p>
                    {activeFiles.length === 0 && (
                      <p className="text-sm text-gray-600 mt-2">
                        No active files found. Upload files in the <Link href="/upload" className="text-blue-600 hover:underline">Upload</Link> section and mark them as active in the <Link href="/view" className="text-blue-600 hover:underline">View</Link> section.
                      </p>
                    )}
                    {activeFiles.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs text-gray-600">Ready to analyze:</p>
                        <ul className="text-xs text-gray-600 mt-1">
                          {activeFiles.slice(0, 3).map(file => (
                            <li key={file.id} className="truncate">• {file.name}</li>
                          ))}
                          {activeFiles.length > 3 && (
                            <li className="text-gray-500">• ... and {activeFiles.length - 3} more</li>
                          )}
                        </ul>
                      </div>
                    )}
                  </div>
                  {activeFiles.length > 0 && (
                    <div className="flex flex-wrap gap-3 justify-center">
                      <button 
                        onClick={() => handleSuggestionClick("Provide a summary of my data")}
                        className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                      >
                        Summarize my data
                      </button>
                      <button 
                        onClick={() => handleSuggestionClick("What trends can you find in my data?")}
                        className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                      >
                        Find trends
                      </button>
                      <button 
                        onClick={() => handleSuggestionClick("What insights can you provide from my data?")}
                        className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                      >
                        Key insights
                      </button>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div>
                  {activeChat?.messages.map((msg) => (
                    <MessageBubble 
                      key={msg.id} 
                      message={msg} 
                      isOwn={msg.sender === 'user'} 
                    />
                  ))}
                  {isLoadingFiles && (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="flex items-start space-x-3 mb-4"
                    >
                      {getAIAvatar()}
                      <div className="bg-gray-200 rounded-2xl rounded-bl-md px-4 py-3 transition-all duration-300 shadow-sm">
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                          <span className="text-sm text-gray-700">Loading active files...</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
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
                  placeholder={activeFiles.length > 0 ? "Ask me anything about your active files..." : "Upload and activate files to start analysis..."}
                  rows="1"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-2">
                  <button 
                    onClick={handleSendMessage}
                    disabled={!message.trim() || activeFiles.length === 0}
                    className="w-8 h-8 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center transform hover:scale-110 active:scale-90 disabled:hover:scale-100 shadow-md hover:shadow-lg"
                  >
                    <svg className="w-4 h-4 transform transition-transform duration-200 hover:translate-x-0.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
                    </svg>
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-600 mt-2 text-center transition-opacity duration-200">
                {activeFiles.length > 0 
                  ? "Press Enter to send, Shift+Enter for new line" 
                  : "Upload and activate files to start analysis"}
              </p>
            </div>
          </motion.div>
        </motion.main>
      </motion.div>
    </div>
  );
}