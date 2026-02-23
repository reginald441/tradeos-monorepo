import React from 'react';
import {
  Bell,
  Search,
  User,
  Moon,
  Sun,
  Wallet,
  TrendingUp,
  TrendingDown,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { useTradingStore } from '@/store/tradingStore';

interface HeaderProps {
  isSidebarCollapsed: boolean;
}

export const Header: React.FC<HeaderProps> = ({ isSidebarCollapsed }) => {
  const user = useAuthStore((state) => state.user);
  const portfolio = useTradingStore((state) => state.portfolio);
  const [isDark, setIsDark] = React.useState(true);
  const [showNotifications, setShowNotifications] = React.useState(false);

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const notifications = [
    { id: 1, title: 'Position Liquidated', message: 'SOL-USD position was liquidated', type: 'danger', time: '2 min ago' },
    { id: 2, title: 'Strategy Alert', message: 'Trend Follower generated a buy signal', type: 'info', time: '15 min ago' },
    { id: 3, title: 'Daily Report', message: 'Your daily P&L report is ready', type: 'success', time: '1 hour ago' },
  ];

  return (
    <header
      className={`fixed top-0 right-0 h-16 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800 z-30 transition-all duration-300 ${
        isSidebarCollapsed ? 'left-16' : 'left-64'
      }`}
    >
      <div className="h-full flex items-center justify-between px-6">
        {/* Search */}
        <div className="flex-1 max-w-md">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search markets, strategies, or positions..."
              className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-10 pr-4 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-4">
          {/* Portfolio Quick View */}
          {portfolio && (
            <div className="hidden md:flex items-center gap-4 px-4 py-2 bg-slate-900/50 rounded-lg border border-slate-800/50">
              <div className="flex items-center gap-2">
                <Wallet className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-400">Balance:</span>
                <span className="text-sm font-mono font-medium text-slate-100">
                  {formatCurrency(portfolio.totalValue)}
                </span>
              </div>
              <div className="w-px h-4 bg-slate-700" />
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-400">24h P&L:</span>
                <span className={`text-sm font-mono font-medium flex items-center gap-1 ${
                  portfolio.realizedPnl24h >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {portfolio.realizedPnl24h >= 0 ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  {portfolio.realizedPnl24h >= 0 ? '+' : ''}
                  {formatCurrency(portfolio.realizedPnl24h)}
                </span>
              </div>
            </div>
          )}

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-all"
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-all"
            >
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
            </button>

            {showNotifications && (
              <div className="absolute right-0 top-full mt-2 w-80 bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden animate-in">
                <div className="px-4 py-3 border-b border-slate-800">
                  <h3 className="text-sm font-semibold text-slate-100">Notifications</h3>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  {notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className="px-4 py-3 hover:bg-slate-800/50 cursor-pointer border-b border-slate-800/50 last:border-0"
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <p className="text-sm font-medium text-slate-200">{notification.title}</p>
                          <p className="text-xs text-slate-500 mt-0.5">{notification.message}</p>
                        </div>
                        <span className="text-xs text-slate-600">{notification.time}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="px-4 py-2 border-t border-slate-800 bg-slate-900/50">
                  <button className="text-xs text-blue-400 hover:text-blue-300 font-medium">
                    View all notifications
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* User Menu */}
          <div className="flex items-center gap-3 pl-4 border-l border-slate-800">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-medium text-slate-100">{user?.name}</p>
              <p className="text-xs text-slate-500">{user?.role}</p>
            </div>
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
              <User className="w-5 h-5 text-white" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
