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

  const getFileIcon = (type) => {
    switch (type.toLowerCase()) {
      case 'csv':
        return (
          <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm0 2h12v10H4V5z"/>
          </svg>
        );
      case 'excel':
        return (
          <svg className="w-5 h-5 text-emerald-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm0 2h12v10H4V5z"/>
          </svg>
        );
      case 'json':
        return (
          <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm0 2h12v10H4V5z"/>
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm0 2h12v10H4V5z"/>
          </svg>
        );
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
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)] bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300`}>
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-full p-1 transition-colors duration-300">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 font-medium text-sm transition-all duration-300 cursor-pointer">
              View
            </div>
          </Link>
        </div>
      </nav>

      {/* Page Header */}
      <div className="px-8 py-4">
        <h1 className="text-3xl font-bold text-center">View Your Data</h1>
        <p className="text-center text-gray-600 dark:text-gray-400 mt-2">
          Manage and explore your uploaded datasets
        </p>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-8 py-8">
        
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">{files.length}</h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm">Total Files</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {files.filter(file => file.isActive).length}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm">Active Files</p>
              </div>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {files.filter(file => !file.isActive).length}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm">Inactive Files</p>
              </div>
              <div className="w-12 h-12 bg-red-100 dark:bg-red-900 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {files.reduce((total, file) => total + parseFloat(file.size), 0).toFixed(1)} MB
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm">Total Size</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m-8 6v8m4-8v8M5 4h14l-1 14H6L5 4z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="flex-1">
            <div className="relative">
              <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent transition-colors duration-300"
              />
            </div>
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent transition-colors duration-300"
          >
            <option value="all">All Files</option>
            <option value="active">Active Only</option>
            <option value="inactive">Inactive Only</option>
          </select>
        </div>
        
        {/* Files Table */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden transition-colors duration-300">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">File</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Size</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Type</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Modified</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
                {filteredFiles.map((file) => (
                  <tr 
                    key={file.id} 
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {getFileIcon(file.type)}
                        <div className="ml-3">
                          <div className="text-sm font-medium text-gray-900 dark:text-gray-100">{file.name}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-300">
                      {file.size}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200">
                        {file.type}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-300">
                      {new Date(file.timestamp).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => toggleFileStatus(file.id)}
                        className={`inline-flex px-3 py-1 rounded-full text-xs font-medium transition-colors duration-200 ${
                          file.isActive
                            ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 hover:bg-green-200 dark:hover:bg-green-800'
                            : 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 hover:bg-red-200 dark:hover:bg-red-800'
                        }`}
                      >
                        {file.isActive ? 'Active' : 'Inactive'}
                      </button>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleViewFile(file)}
                          className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors duration-200"
                        >
                          View
                        </button>
                        <button
                          onClick={() => handleDeleteFile(file.id)}
                          className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 transition-colors duration-200"
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
            <div className="text-center py-12">
              <svg className="w-12 h-12 mx-auto text-gray-400 dark:text-gray-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-gray-500 dark:text-gray-400">No files found matching your criteria</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
