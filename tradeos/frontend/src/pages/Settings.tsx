import React, { useState } from 'react';
import {
  User,
  Mail,
  Lock,
  Bell,
  Shield,
  Key,
  Globe,
  Moon,
  Sun,
  Smartphone,
  Save,
  Trash2,
  Copy,
  Eye,
  EyeOff,
  Plus,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input, Select } from '@/components/ui/Input';
import type { ApiKey } from '@/types';

// Mock API keys
const mockApiKeys: ApiKey[] = [
  {
    id: 'key-1',
    name: 'Trading Bot',
    key: 'tr_live_****************************abcd',
    permissions: ['read', 'trade'],
    createdAt: '2024-01-15T10:00:00Z',
    lastUsedAt: '2024-01-17T14:30:00Z',
  },
  {
    id: 'key-2',
    name: 'Data Analysis',
    key: 'tr_live_****************************efgh',
    permissions: ['read'],
    createdAt: '2024-01-10T08:00:00Z',
    lastUsedAt: '2024-01-16T09:15:00Z',
  },
];

export const Settings: React.FC = () => {
  const user = useAuthStore((state) => state.user);
  const updateUser = useAuthStore((state) => state.updateUser);
  
  const [activeTab, setActiveTab] = useState<'profile' | 'security' | 'notifications' | 'api'>('profile');
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>(mockApiKeys);
  const [showKey, setShowKey] = useState<string | null>(null);
  const [showCreateKeyModal, setShowCreateKeyModal] = useState(false);

  const [profileForm, setProfileForm] = useState({
    name: user?.name || '',
    email: user?.email || '',
    timezone: 'UTC',
    language: 'en',
  });

  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [notificationSettings, setNotificationSettings] = useState({
    emailAlerts: true,
    pushNotifications: true,
    tradeNotifications: true,
    strategyAlerts: true,
    riskAlerts: true,
    marketingEmails: false,
  });

  const handleSaveProfile = () => {
    updateUser({ name: profileForm.name });
  };

  const handleChangePassword = () => {
    // Implement password change
  };

  const handleCreateApiKey = () => {
    const newKey: ApiKey = {
      id: `key-${Date.now()}`,
      name: 'New API Key',
      key: `tr_live_${'*'.repeat(28)}${Math.random().toString(36).substr(2, 4)}`,
      permissions: ['read'],
      createdAt: new Date().toISOString(),
    };
    setApiKeys([newKey, ...apiKeys]);
    setShowCreateKeyModal(false);
  };

  const handleDeleteApiKey = (id: string) => {
    setApiKeys(apiKeys.filter(k => k.id !== id));
  };

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'api', label: 'API Keys', icon: Key },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Settings</h1>
        <p className="text-slate-400 mt-1">Manage your account and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Tabs */}
        <Card className="lg:col-span-1 h-fit">
          <CardContent className="p-2">
            <nav className="space-y-1">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all ${
                      activeTab === tab.id
                        ? 'bg-blue-500/10 text-blue-400'
                        : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </CardContent>
        </Card>

        {/* Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <Card>
              <CardHeader title="Profile Information" />
              <CardContent>
                <div className="space-y-6">
                  <div className="flex items-center gap-6">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
                      <User className="w-10 h-10 text-white" />
                    </div>
                    <div>
                      <Button variant="secondary" size="sm">Change Avatar</Button>
                      <p className="text-xs text-slate-500 mt-2">JPG, PNG or GIF. Max 2MB.</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input
                      label="Full Name"
                      value={profileForm.name}
                      onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
                      leftIcon={<User className="w-4 h-4" />}
                    />
                    <Input
                      label="Email"
                      type="email"
                      value={profileForm.email}
                      onChange={(e) => setProfileForm({ ...profileForm, email: e.target.value })}
                      leftIcon={<Mail className="w-4 h-4" />}
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Select
                      label="Timezone"
                      options={[
                        { value: 'UTC', label: 'UTC' },
                        { value: 'EST', label: 'Eastern Time (EST)' },
                        { value: 'PST', label: 'Pacific Time (PST)' },
                        { value: 'GMT', label: 'Greenwich Mean Time (GMT)' },
                      ]}
                      value={profileForm.timezone}
                      onChange={(e) => setProfileForm({ ...profileForm, timezone: e.target.value })}
                    />
                    <Select
                      label="Language"
                      options={[
                        { value: 'en', label: 'English' },
                        { value: 'es', label: 'Spanish' },
                        { value: 'fr', label: 'French' },
                        { value: 'de', label: 'German' },
                      ]}
                      value={profileForm.language}
                      onChange={(e) => setProfileForm({ ...profileForm, language: e.target.value })}
                    />
                  </div>

                  <div className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg">
                    <div className="flex items-center gap-3">
                      {isDarkMode ? <Moon className="w-5 h-5 text-slate-400" /> : <Sun className="w-5 h-5 text-slate-400" />}
                      <div>
                        <p className="font-medium text-slate-200">Dark Mode</p>
                        <p className="text-sm text-slate-500">Toggle between light and dark themes</p>
                      </div>
                    </div>
                    <button
                      onClick={() => setIsDarkMode(!isDarkMode)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        isDarkMode ? 'bg-blue-500' : 'bg-slate-700'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        isDarkMode ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </div>

                  <div className="flex justify-end">
                    <Button leftIcon={<Save className="w-4 h-4" />} onClick={handleSaveProfile}>
                      Save Changes
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <Card>
              <CardHeader title="Change Password" />
              <CardContent>
                <div className="space-y-4 max-w-md">
                  <Input
                    label="Current Password"
                    type="password"
                    value={passwordForm.currentPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, currentPassword: e.target.value })}
                    leftIcon={<Lock className="w-4 h-4" />}
                  />
                  <Input
                    label="New Password"
                    type="password"
                    value={passwordForm.newPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                    leftIcon={<Lock className="w-4 h-4" />}
                  />
                  <Input
                    label="Confirm New Password"
                    type="password"
                    value={passwordForm.confirmPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                    leftIcon={<Lock className="w-4 h-4" />}
                  />
                  <Button onClick={handleChangePassword}>Update Password</Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Notifications Tab */}
          {activeTab === 'notifications' && (
            <Card>
              <CardHeader title="Notification Preferences" />
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(notificationSettings).map(([key, value]) => {
                    const labels: Record<string, string> = {
                      emailAlerts: 'Email Alerts',
                      pushNotifications: 'Push Notifications',
                      tradeNotifications: 'Trade Notifications',
                      strategyAlerts: 'Strategy Alerts',
                      riskAlerts: 'Risk Alerts',
                      marketingEmails: 'Marketing Emails',
                    };
                    
                    return (
                      <div key={key} className="flex items-center justify-between py-3 border-b border-slate-800/50 last:border-0">
                        <div>
                          <p className="font-medium text-slate-200">{labels[key]}</p>
                          <p className="text-sm text-slate-500">
                            {key === 'emailAlerts' && 'Receive important updates via email'}
                            {key === 'pushNotifications' && 'Get push notifications in your browser'}
                            {key === 'tradeNotifications' && 'Notify when trades are executed'}
                            {key === 'strategyAlerts' && 'Get alerts for strategy events'}
                            {key === 'riskAlerts' && 'Receive risk threshold notifications'}
                            {key === 'marketingEmails' && 'Receive product updates and offers'}
                          </p>
                        </div>
                        <button
                          onClick={() => setNotificationSettings({ ...notificationSettings, [key]: !value })}
                          className={`w-12 h-6 rounded-full transition-colors ${
                            value ? 'bg-blue-500' : 'bg-slate-700'
                          }`}
                        >
                          <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                            value ? 'translate-x-6' : 'translate-x-0.5'
                          }`} />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* API Keys Tab */}
          {activeTab === 'api' && (
            <Card>
              <CardHeader 
                title="API Keys" 
                subtitle="Manage your API keys for programmatic access"
                action={
                  <Button 
                    size="sm" 
                    leftIcon={<Plus className="w-4 h-4" />}
                    onClick={() => setShowCreateKeyModal(true)}
                  >
                    Create Key
                  </Button>
                }
              />
              <CardContent>
                <div className="space-y-4">
                  {apiKeys.map((key) => (
                    <div key={key.id} className="p-4 bg-slate-900/50 rounded-lg border border-slate-800/50">
                      <div className="flex items-start justify-between">
                        <div>
                          <p className="font-medium text-slate-200">{key.name}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <code className="text-sm text-slate-500 font-mono">
                              {showKey === key.id ? key.key : key.key.replace(/\*/g, '•')}
                            </code>
                            <button
                              onClick={() => setShowKey(showKey === key.id ? null : key.id)}
                              className="text-slate-500 hover:text-slate-300"
                            >
                              {showKey === key.id ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </button>
                            <button
                              onClick={() => navigator.clipboard.writeText(key.key)}
                              className="text-slate-500 hover:text-slate-300"
                            >
                              <Copy className="w-4 h-4" />
                            </button>
                          </div>
                          <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                            <span>Created: {new Date(key.createdAt).toLocaleDateString()}</span>
                            {key.lastUsedAt && (
                              <span>Last used: {new Date(key.lastUsedAt).toLocaleDateString()}</span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex gap-1">
                            {key.permissions.map((perm) => (
                              <span 
                                key={perm} 
                                className="px-2 py-0.5 bg-slate-800 text-slate-400 text-xs rounded"
                              >
                                {perm}
                              </span>
                            ))}
                          </div>
                          <button
                            onClick={() => handleDeleteApiKey(key.id)}
                            className="p-2 text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Create API Key Modal */}
      {showCreateKeyModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-slate-100 mb-4">Create API Key</h2>
            <div className="space-y-4">
              <Input label="Key Name" placeholder="e.g., Trading Bot" />
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Permissions</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded border-slate-700 bg-slate-800 text-blue-500" defaultChecked />
                    <span className="text-sm text-slate-400">Read</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded border-slate-700 bg-slate-800 text-blue-500" />
                    <span className="text-sm text-slate-400">Trade</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded border-slate-700 bg-slate-800 text-blue-500" />
                    <span className="text-sm text-slate-400">Withdraw</span>
                  </label>
                </div>
              </div>
              <div className="flex justify-end gap-3 pt-4">
                <Button variant="secondary" onClick={() => setShowCreateKeyModal(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateApiKey}>Create Key</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
