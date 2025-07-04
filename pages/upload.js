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

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      // Handle file upload logic here
      console.log("Uploading file:", selectedFile.name);
    }
  };

  return (
    <div
      className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)]`}
    >
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
        <div className="flex bg-black/[.05] dark:bg-white/[.06] rounded-full p-1">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-black/[.05] dark:hover:bg-white/[.06] font-medium text-sm transition-colors cursor-pointer">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full bg-foreground text-background font-medium text-sm transition-colors cursor-pointer">
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

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center min-h-[calc(100vh-120px)] px-8">
        <div className="max-w-2xl w-full">
          <h1 className="text-3xl font-bold text-center mb-8">Upload Your Data</h1>
          
          {/* File Upload Area */}
          <div
            className={`border-2 rounded-lg p-12 text-center transition-colors ${
              selectedFile
                ? 'border-solid border-green-400 bg-green-50 dark:bg-green-950'
                : isDragging
                ? 'border-dashed border-blue-500 bg-blue-50 dark:bg-blue-950'
                : 'border-dashed border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              <div className="flex justify-center">
                <svg
                  className="w-12 h-12 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              
              <div>
                {selectedFile && (
                  <p className="text-lg font-medium mb-2">
                    {selectedFile.name}
                  </p>
                )}
                <p className="text-lg font-medium mb-2">
                  Drop your file here
                </p>
                <p className="text-gray-600 dark:text-gray-300">
                  or
                </p>
              </div>
              
              <input
                type="file"
                onChange={handleFileSelect}
                className="hidden"
                id="fileInput"
                accept=".csv,.xlsx,.xls,.json,.txt"
              />
              
              <label
                htmlFor="fileInput"
                className="inline-block px-6 py-3 bg-black/[.05] dark:bg-white/[.06] rounded-lg font-medium cursor-pointer hover:bg-black/[.08] dark:hover:bg-white/[.08] transition-colors"
              >
                Select File
              </label>
            </div>
          </div>
          
          {/* File Information */}
          {selectedFile && (
            <div className="mt-6 p-4 bg-black/[.05] dark:bg-white/[.06] rounded-lg">
              <h3 className="font-medium mb-2">Selected File:</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Name: {selectedFile.name}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Type: {selectedFile.type || 'Unknown'}
              </p>
            </div>
          )}
          
          {/* Upload Button */}
          <div className="mt-8 text-center">
            <button
              onClick={handleUpload}
              disabled={!selectedFile}
              className={`px-8 py-3 rounded-lg font-medium transition-colors ${
                selectedFile
                  ? 'bg-foreground text-background hover:bg-[#383838] dark:hover:bg-[#ccc]'
                  : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
              }`}
            >
              Upload File
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
