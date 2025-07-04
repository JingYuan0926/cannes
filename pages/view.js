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

export default function View() {
  const [files, setFiles] = useState([
    {
      id: 1,
      name: "File 1",
      timestamp: "4 July 2025 00:00:00",
      isActive: true,
    },
    {
      id: 2,
      name: "File 2",
      timestamp: "4 July 2025 01:00:00",
      isActive: false,
    },
    {
      id: 3,
      name: "File 3",
      timestamp: "4 July 2025 02:00:00",
      isActive: true,
    },
  ]);

  const toggleFileStatus = (id) => {
    setFiles(files.map(file => 
      file.id === id ? { ...file, isActive: !file.isActive } : file
    ));
  };

  const handleViewFile = (file) => {
    // Handle file viewing logic here
    console.log("Viewing file:", file.name);
    // This would typically open a modal or navigate to a file viewer
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
            <div className="px-6 py-2 rounded-full hover:bg-black/[.05] dark:hover:bg-white/[.06] font-medium text-sm transition-colors cursor-pointer">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full bg-foreground text-background font-medium text-sm transition-colors cursor-pointer">
              View
            </div>
          </Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex flex-col items-center min-h-[calc(100vh-120px)] px-8 py-8">
        <div className="max-w-4xl w-full">
          <h1 className="text-3xl font-bold text-center mb-8">View Your Data</h1>
          
          {/* Files Table */}
          <div className="bg-black/[.05] dark:bg-white/[.06] rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-black/[.08] dark:bg-white/[.08]">
                  <tr>
                    <th className="px-6 py-4 text-left font-medium">File Name</th>
                    <th className="px-6 py-4 text-left font-medium">Timestamp</th>
                    <th className="px-6 py-4 text-left font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((file) => (
                    <tr 
                      key={file.id} 
                      className="border-t border-gray-200 dark:border-gray-700 hover:bg-black/[.02] dark:hover:bg-white/[.02] cursor-pointer transition-colors"
                      onClick={() => handleViewFile(file)}
                    >
                      <td className="px-6 py-4 font-medium">{file.name}</td>
                      <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                        {file.timestamp}
                      </td>
                      <td className="px-6 py-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFileStatus(file.id);
                          }}
                          className={`px-3 py-1 rounded-full text-sm font-medium transition-colors cursor-pointer hover:opacity-80 ${
                            file.isActive
                              ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200'
                              : 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
                          }`}
                        >
                          {file.isActive ? 'Active' : 'Inactive'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Summary Stats */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-black/[.05] dark:bg-white/[.06] rounded-lg p-6 text-center">
              <h3 className="text-2xl font-bold">{files.length}</h3>
              <p className="text-gray-600 dark:text-gray-300">Total Files</p>
            </div>
            <div className="bg-black/[.05] dark:bg-white/[.06] rounded-lg p-6 text-center">
              <h3 className="text-2xl font-bold text-green-600">
                {files.filter(file => file.isActive).length}
              </h3>
              <p className="text-gray-600 dark:text-gray-300">Active Files</p>
            </div>
            <div className="bg-black/[.05] dark:bg-white/[.06] rounded-lg p-6 text-center">
              <h3 className="text-2xl font-bold text-red-600">
                {files.filter(file => !file.isActive).length}
              </h3>
              <p className="text-gray-600 dark:text-gray-300">Inactive Files</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
