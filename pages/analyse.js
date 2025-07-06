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
  const [activeReports, setActiveReports] = useState([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages, isAiTyping]);

  // Load active files and reports from localStorage
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
    
    const loadActiveReports = () => {
      try {
        const storedReports = JSON.parse(localStorage.getItem('analysisReports') || '[]');
        console.log('All stored reports:', storedReports);
        const activeReports = storedReports.filter(report => report.isActive);
        console.log('Active reports found:', activeReports);
        setActiveReports(activeReports);
      } catch (error) {
        console.error('Failed to load active reports:', error);
        setActiveReports([]);
      }
    };
    
    loadActiveFiles();
    loadActiveReports();
    
    // Listen for storage changes
    const handleStorageChange = () => {
      loadActiveFiles();
      loadActiveReports();
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
    if (activeFiles.length === 0 && activeReports.length === 0) {
      console.log('No active files or reports to load');
      return '';
    }

    console.log(`Loading content for ${activeFiles.length} active files and ${activeReports.length} active reports:`, { activeFiles, activeReports });
    setIsLoadingFiles(true);
    let combinedContent = '';
    
    try {
      // Load active files
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
      
      // Load active reports
      for (const report of activeReports) {
        try {
          console.log(`Loading analysis report for: ${report.fileName}`);
          
          // Load analysis results from Walrus or local storage
          let analysisResults = null;
          if (report.analysisResultsBlobId) {
            try {
              const response = await fetch(`https://publisher-devnet.walrus.space/v1/${report.analysisResultsBlobId}`);
              if (response.ok) {
                const analysisResultsJson = await response.text();
                analysisResults = JSON.parse(analysisResultsJson);
              }
            } catch (error) {
              console.error(`Failed to load analysis results from Walrus for ${report.fileName}:`, error);
            }
          }
          
          // Fallback to local storage
          if (!analysisResults && report.analysisResults) {
            analysisResults = report.analysisResults;
          }
          
          if (analysisResults) {
            // Create a summary of the analysis results
            let reportSummary = `\n\n--- Analysis Report: ${report.fileName} ---\n`;
            reportSummary += `Analysis Goal: ${report.analysisGoal}\n`;
            reportSummary += `Analysis Date: ${new Date(report.timestamp).toLocaleString()}\n`;
            reportSummary += `Status: ${report.status}\n\n`;
            
            // Add key insights from the analysis
            if (analysisResults.ml?.aiInsights?.keyInsights) {
              reportSummary += `Key Insights:\n${analysisResults.ml.aiInsights.keyInsights.join('\n')}\n\n`;
            }
            
            // Add data summary
            if (analysisResults.etl?.summary) {
              reportSummary += `Data Summary:\n${JSON.stringify(analysisResults.etl.summary, null, 2)}\n\n`;
            }
            
            // Add EDA summary
            if (analysisResults.eda?.analysis?.summary) {
              reportSummary += `EDA Summary:\n${JSON.stringify(analysisResults.eda.analysis.summary, null, 2)}\n\n`;
            }
            
            // Add ML analysis summary
            if (analysisResults.ml?.results?.summary) {
              reportSummary += `ML Analysis Summary:\n${JSON.stringify(analysisResults.ml.results.summary, null, 2)}\n\n`;
            }
            
            combinedContent += reportSummary;
            console.log(`Added analysis report for ${report.fileName}, length: ${reportSummary.length}`);
          } else {
            combinedContent += `\n\n--- Analysis Report: ${report.fileName} ---\n[Error loading analysis results]`;
            console.log(`Failed to load analysis results for ${report.fileName}`);
          }
        } catch (error) {
          console.error(`Failed to load analysis report for ${report.fileName}:`, error);
          combinedContent += `\n\n--- Analysis Report: ${report.fileName} ---\n[Error loading analysis report: ${error.message}]`;
        }
      }
    } catch (error) {
      console.error('Error loading active files and reports:', error);
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
    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-medium shadow-md transform transition-all duration-200 hover:scale-110 hover:shadow-lg">
      U
    </div>
  );

  const getAIAvatar = () => (
    <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white text-sm font-medium shadow-md">
      AI
    </div>
  );

  const TypingIndicator = () => (
    <div className="flex items-center space-x-1 p-3">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl rounded-bl-md px-4 py-3 transition-all duration-300 shadow-sm hover:shadow-md">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );

  const MessageBubble = ({ message, isOwn }) => (
    <div className={`flex items-start space-x-2 p-3 ${isOwn ? 'flex-row-reverse space-x-reverse' : ''}`}>
      <div className={`flex-shrink-0 ${isOwn ? 'order-2' : 'order-1'}`}>
        {isOwn ? getUserAvatar() : getAIAvatar()}
      </div>
      <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl transition-all duration-300 ${
        isOwn 
          ? 'bg-blue-600 text-white rounded-br-md shadow-md hover:bg-blue-700' 
          : 'bg-white/80 backdrop-blur-sm text-slate-900 rounded-bl-md shadow-sm hover:shadow-md border border-blue-200'
      }`}>
        <div className="whitespace-pre-wrap break-words">
          {isOwn ? (
            message.text
          ) : (
            <ReactMarkdown 
              remarkPlugins={[remarkGfm, remarkBreaks]}
              rehypePlugins={[rehypeRaw]}
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                code: ({ node, inline, className, children, ...props }) => {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline ? (
                    <pre className="bg-slate-100 p-2 rounded text-sm overflow-x-auto my-2">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  ) : (
                    <code className="bg-slate-100 px-1 py-0.5 rounded text-sm" {...props}>
                      {children}
                    </code>
                  );
                },
                ul: ({ children }) => <ul className="list-disc list-inside mb-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside mb-2">{children}</ol>,
                li: ({ children }) => <li className="mb-1">{children}</li>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-blue-300 pl-4 italic my-2">{children}</blockquote>
                ),
                h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                h2: ({ children }) => <h2 className="text-md font-semibold mb-2">{children}</h2>,
                h3: ({ children }) => <h3 className="text-sm font-medium mb-1">{children}</h3>,
              }}
            >
              {message.text}
            </ReactMarkdown>
          )}
        </div>
        <span className="text-xs text-slate-600"></span>
      </div>
    </div>
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
    <div className="h-screen font-montserrat bg-gradient-to-br from-blue-50 to-indigo-100 text-slate-900 transition-colors duration-300 overflow-hidden flex flex-col">
      <style jsx>{`
        .react-markdown a {
          color: #2563eb !important;
          text-decoration: underline;
          cursor: pointer;
        }
        .react-markdown a:hover {
          color: #3b82f6 !important;
        }
      `}</style>
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative flex justify-center pt-8 pb-4 px-8 flex-shrink-0"
      >
        <div className="flex bg-white/80 backdrop-blur-sm rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl border border-blue-200">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full bg-blue-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              View
            </div>
          </Link>
          <Link href="/subscribe">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Subscribe
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
        className="px-8 py-2 flex-shrink-0"
      >
        <h1 className="text-3xl font-bold text-center text-slate-800 transform transition-all duration-300">Analyse Your Data</h1>
        <p className="text-center text-slate-600 mt-1">Chat with our AI to analyze and understand your data better</p>
      </motion.div>

      {/* Main Layout */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.2 }}
        className="flex flex-1 min-h-0 relative"
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
              className="absolute left-0 top-0 bottom-0 w-64 bg-white/90 backdrop-blur-sm flex flex-col transition-all duration-300 shadow-xl border-r-2 border-blue-200 z-20"
            >
              {/* Sidebar Header with Close Button */}
              <div className="flex items-center justify-between p-4 pb-2 border-b border-blue-200">
                <h3 className="font-medium text-sm text-slate-800">Chat History</h3>
                <button
                  onClick={toggleSidebar}
                  className="p-2 rounded-lg bg-blue-100 hover:bg-blue-200 transition-all duration-300 ease-in-out shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95 flex items-center justify-center"
                  title="Hide Chat History"
                >
                  <svg className="w-4 h-4 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
              </div>
              
              {/* Chat History */}
              <div className="flex-1 min-h-0 flex flex-col">
                <div className="flex-1 px-4 pb-4 overflow-y-auto">
                  <div className="space-y-2">
                    {chats.length === 0 ? (
                      <div className="text-center py-8">
                        <p className="text-sm text-slate-500 mb-4">No chats yet</p>
                        <p className="text-xs text-slate-400">Start a conversation to create your first chat</p>
                      </div>
                    ) : (
                      chats.map((chat, index) => (
                        <motion.div
                          key={chat.id}
                          variants={itemVariants}
                          onClick={() => handleChatSelect(chat)}
                          className={`p-3 rounded-lg cursor-pointer transition-all duration-200 transform hover:scale-[1.01] hover:shadow-md active:scale-95 ${
                            activeChat?.id === chat.id
                              ? 'bg-blue-600 text-white shadow-lg scale-[1.01]'
                              : 'hover:bg-blue-100 text-slate-800'
                          }`}
                        >
                          <div className="flex justify-between items-center">
                            <span className="font-medium truncate flex-1 mr-2">{chat.name}</span>
                            {chat.messages.length > 0 && (
                              <span className={`text-xs opacity-75 px-2 py-1 rounded-full transition-all duration-200 flex-shrink-0 ${
                                activeChat?.id === chat.id ? 'bg-white/20 hover:bg-white/30' : 'bg-blue-200 hover:bg-blue-300'
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
                  className="w-full py-3 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 transition-all duration-200 flex items-center justify-center gap-2 transform hover:scale-[1.01] active:scale-95 shadow-md hover:shadow-lg"
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
          className="flex-1 flex flex-col bg-white/30 backdrop-blur-sm min-h-0 w-full"
        >
          {/* Sidebar Toggle Button */}
          <div className="p-3 flex justify-start">
            <button
              onClick={toggleSidebar}
              className="p-3 rounded-2xl bg-white/80 backdrop-blur-sm hover:bg-blue-100 transition-all duration-300 ease-in-out shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95 flex items-center justify-center z-10 border border-blue-200"
              title={sidebarVisible ? "Hide Chat History" : "Show Chat History"}
            >
              {sidebarVisible ? (
                <svg className="w-5 h-5 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
            </button>
          </div>
          
          {/* Chat Area */}
          <div className="flex-1 p-6 overflow-y-auto min-h-0">
            <div className="max-w-4xl mx-auto">
              {!activeChat || activeChat.messages.length === 0 ? (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="text-center py-8"
                >
                  <h3 className="text-lg font-medium mb-2 text-slate-800 mt-2">Start Your Analysis</h3>
                  <p className="text-slate-600 mb-3">
                    Ask me anything about your data and I'll help you discover insights.
                  </p>
                  <div className="mb-6 p-4 bg-gray-100 rounded-lg">
                    <p className="text-sm text-gray-700">
                      <strong>Active Data:</strong> {activeFiles.length} file{activeFiles.length !== 1 ? 's' : ''} and {activeReports.length} analysis report{activeReports.length !== 1 ? 's' : ''} ready for analysis
                    </p>
                    {activeFiles.length === 0 && activeReports.length === 0 && (
                      <p className="text-sm text-gray-600 mt-2">
                        No active files or reports found. Upload files in the <Link href="/upload" className="text-blue-600 hover:underline">Upload</Link> section and mark them as active in the <Link href="/view" className="text-blue-600 hover:underline">View</Link> section.
                      </p>
                    )}
                    {(activeFiles.length > 0 || activeReports.length > 0) && (
                      <div className="mt-2">
                        <p className="text-xs text-gray-600">Ready to analyze:</p>
                        <ul className="text-xs text-gray-600 mt-1">
                          {activeFiles.slice(0, 2).map(file => (
                            <li key={file.id} className="truncate">• File: {file.name}</li>
                          ))}
                          {activeReports.slice(0, 2).map(report => (
                            <li key={report.id} className="truncate">• Report: {report.fileName}</li>
                          ))}
                          {(activeFiles.length + activeReports.length) > 4 && (
                            <li className="text-gray-500">• ... and {(activeFiles.length + activeReports.length) - 4} more</li>
                          )}
                        </ul>
                      </div>
                    )}
                  </div>
                  {(activeFiles.length > 0 || activeReports.length > 0) && (
                    <div className="flex flex-wrap gap-3 justify-center">
                      <button 
                        onClick={() => handleSuggestionClick("Provide a summary of my data")}
                        className="px-3 py-2 bg-blue-100 text-slate-800 rounded-lg hover:bg-blue-200 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                      >
                        Summarize my data
                      </button>
                      <button 
                        onClick={() => handleSuggestionClick("What trends can you find in my data?")}
                        className="px-3 py-2 bg-blue-100 text-slate-800 rounded-lg hover:bg-blue-200 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
                      >
                        Find trends
                      </button>
                      <button 
                        onClick={() => handleSuggestionClick("What insights can you provide from my data?")}
                        className="px-3 py-2 bg-blue-100 text-slate-800 rounded-lg hover:bg-blue-200 transition-all duration-200 text-sm hover:shadow-lg transform hover:scale-105 active:scale-95 hover:-translate-y-1"
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
                      <div className="bg-blue-100 rounded-2xl rounded-bl-md px-4 py-3 transition-all duration-300 shadow-sm">
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                          <span className="text-sm text-gray-700">Loading active files and reports...</span>
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
            className="p-6 pt-0 flex-shrink-0"
          >
            <div className="relative max-w-4xl mx-auto">
              <div className="relative bg-white/80 backdrop-blur-sm rounded-2xl transition-all duration-300 focus-within:border-blue-500 focus-within:shadow-xl focus-within:scale-[1.01] shadow-md border border-blue-200">
                <textarea
                  value={message}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  className="w-full h-12 p-4 pr-12 bg-transparent resize-none focus:outline-none placeholder-gray-600 text-black transition-all duration-200 overflow-hidden"
                  placeholder={(activeFiles.length > 0 || activeReports.length > 0) ? "Ask me anything about your active files and reports..." : "Upload and activate files/reports to start analysis..."}
                  rows="1"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-2">
                  <button 
                    onClick={handleSendMessage}
                    disabled={!message.trim() || (activeFiles.length === 0 && activeReports.length === 0)}
                    className="w-8 h-8 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center transform hover:scale-110 active:scale-90 disabled:hover:scale-100 shadow-md hover:shadow-lg"
                  >
                    <svg className="w-4 h-4 transform transition-transform duration-200 hover:translate-x-0.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
                    </svg>
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-600 mt-2 text-center transition-opacity duration-200">
                {(activeFiles.length > 0 || activeReports.length > 0)
                  ? "Press Enter to send, Shift+Enter for new line" 
                  : "Upload and activate files/reports to start analysis"}
              </p>
            </div>
          </motion.div>
        </motion.main>
      </motion.div>
    </div>
  );
}