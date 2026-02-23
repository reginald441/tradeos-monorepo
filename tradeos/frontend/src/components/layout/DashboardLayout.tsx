import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const DashboardLayout: React.FC = () => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  return (
    <div className="min-h-screen bg-slate-950">
      <Sidebar isCollapsed={isSidebarCollapsed} onToggle={toggleSidebar} />
      <Header isSidebarCollapsed={isSidebarCollapsed} />
      
      <main
        className={`pt-16 transition-all duration-300 ${
          isSidebarCollapsed ? 'pl-16' : 'pl-64'
        }`}
      >
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
};
