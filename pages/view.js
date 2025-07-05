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

export default function View() {
  const [files, setFiles] = useState([
    {
      id: 1,
      name: "Sales_Data_Q4_2024.csv",
      timestamp: "4 July 2025 00:00:00",
      size: "2.4 MB",
      type: "CSV",
      isActive: true,
    },
    {
      id: 2,
      name: "Customer_Analytics.xlsx",
      timestamp: "4 July 2025 01:00:00",
      size: "5.7 MB",
      type: "Excel",
      isActive: false,
    },
    {
      id: 3,
      name: "Marketing_Metrics.json",
      timestamp: "4 July 2025 02:00:00",
      size: "1.2 MB",
      type: "JSON",
      isActive: true,
    },
    {
      id: 4,
      name: "Product_Performance.csv",
      timestamp: "3 July 2025 18:30:00",
      size: "3.8 MB",
      type: "CSV",
      isActive: true,
    },
    {
      id: 5,
      name: "User_Behavior.txt",
      timestamp: "3 July 2025 15:45:00",
      size: "890 KB",
      type: "Text",
      isActive: false,
    },
  ]);

  const [darkMode, setDarkMode] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  // Dark mode persistence
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedDarkMode !== null) {
      setDarkMode(JSON.parse(savedDarkMode));
    } else {
      setDarkMode(false); // Default to light mode instead of system preference
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

  const toggleFileStatus = (id) => {
    setFiles(files.map(file => 
      file.id === id ? { ...file, isActive: !file.isActive } : file
    ));
  };

  const handleViewFile = (file) => {
    console.log("Viewing file:", file.name);
    // This would typically open a modal or navigate to a file viewer
  };

  const handleDeleteFile = (id) => {
    if (confirm("Are you sure you want to delete this file?")) {
      setFiles(files.filter(file => file.id !== id));
    }
  };



  const filteredFiles = files.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === "all" || 
                         (filterStatus === "active" && file.isActive) ||
                         (filterStatus === "inactive" && !file.isActive);
    return matchesSearch && matchesFilter;
  });

  return (
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)] bg-white text-gray-900 dark:text-gray-100 transition-colors duration-300`}>
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
              View
            </div>
          </Link>
        </div>
      </nav>

      {/* Page Header */}
      <div className="px-8 py-4">
        <h1 className="text-3xl font-bold text-center text-black transform transition-all duration-300">View Your Data</h1>
        <p className="text-center text-gray-600 dark:text-gray-400 mt-2 transition-opacity duration-200">
          Manage and explore your uploaded datasets
        </p>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-8 py-8">
        
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md animate-slideInUp">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-black counter-animation">{files.length}</h3>
                <p className="text-black text-sm">Total Files</p>
              </div>
              <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md animate-slideInUp" style={{ animationDelay: '0.1s' }}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-green-600 counter-animation">
                  {files.filter(file => file.isActive).length}
                </h3>
                <p className="text-black text-sm">Active Files</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md animate-slideInUp" style={{ animationDelay: '0.2s' }}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-red-600 counter-animation">
                  {files.filter(file => !file.isActive).length}
                </h3>
                <p className="text-black text-sm">Inactive Files</p>
              </div>
              <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md animate-slideInUp" style={{ animationDelay: '0.3s' }}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-purple-600 counter-animation">
                  {files.reduce((total, file) => total + parseFloat(file.size), 0).toFixed(1)} MB
                </h3>
                <p className="text-black text-sm">Total Size</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m-8 6v8m4-8v8M5 4h14l-1 14H6L5 4z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="flex-1">
            <div className="relative group">
              <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-600 transition-all duration-200 group-focus-within:text-gray-800 group-focus-within:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-gray-200 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all duration-300 hover:shadow-md focus:shadow-lg transform focus:scale-[1.02] text-black placeholder-gray-600"
              />
            </div>
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-3 bg-gray-200 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all duration-300 hover:shadow-md focus:shadow-lg transform hover:scale-105 focus:scale-105 text-black"
          >
            <option value="all">All Files</option>
            <option value="active">Active Only</option>
            <option value="inactive">Inactive Only</option>
          </select>
        </div>
        
        {/* Files Table */}
        <div className="bg-gray-200 rounded-2xl border border-gray-300 overflow-hidden transition-all duration-300 shadow-lg hover:shadow-xl animate-slideInUp">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-300 transition-colors duration-300">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200">File</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200">Size</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200">Type</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200">Modified</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200">Status</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-300">
                {filteredFiles.map((file, index) => (
                  <tr 
                    key={file.id} 
                    className="hover:bg-gray-300 transition-all duration-200 group hover:shadow-md animate-slideInUp"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-black group-hover:text-gray-800 transition-colors duration-200">
                        {file.name}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-black group-hover:text-gray-800 transition-colors duration-200">
                      {file.size}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-gray-100 text-gray-800 transition-all duration-200 group-hover:scale-105 group-hover:bg-gray-50">
                        {file.type}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-black group-hover:text-gray-800 transition-colors duration-200">
                      {new Date(file.timestamp).toLocaleDateString('en-GB', {
                        day: '2-digit',
                        month: '2-digit', 
                        year: 'numeric'
                      })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => toggleFileStatus(file.id)}
                        className={`inline-flex px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 ${
                          file.isActive
                            ? 'bg-green-100 text-green-800 hover:bg-green-200 hover:shadow-md'
                            : 'bg-red-100 text-red-800 hover:bg-red-200 hover:shadow-md'
                        }`}
                      >
                        {file.isActive ? 'Active' : 'Inactive'}
                      </button>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleViewFile(file)}
                          className="text-gray-600 hover:text-gray-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
                        >
                          View
                        </button>
                        <button
                          onClick={() => handleDeleteFile(file.id)}
                          className="text-red-600 hover:text-red-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredFiles.length === 0 && (
            <div className="text-center py-12 animate-fadeIn">
              <svg className="w-12 h-12 mx-auto text-gray-600 mb-4 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-black text-lg">No files found matching your criteria</p>
            </div>
          )}
        </div>
      </main>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes countUp {
          from {
            opacity: 0;
            transform: scale(0.5);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }

        .animate-slideInUp {
          animation: slideInUp 0.4s ease-out;
        }

        .counter-animation {
          animation: countUp 0.6s ease-out;
        }
      `}</style>
    </div>
  );
}
