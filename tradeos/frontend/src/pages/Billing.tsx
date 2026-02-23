import React, { useState } from 'react';
import {
  CreditCard,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  Zap,
  Shield,
  Building2,
  ChevronRight,
  Star,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import type { Subscription, Invoice } from '@/types';

// Mock data
const mockSubscription: Subscription = {
  id: 'sub-1',
  plan: 'pro',
  status: 'active',
  currentPeriodStart: '2024-01-01T00:00:00Z',
  currentPeriodEnd: '2024-02-01T00:00:00Z',
  cancelAtPeriodEnd: false,
  features: [
    'Unlimited strategies',
    'Advanced backtesting',
    'Real-time data',
    'Priority support',
    'API access',
  ],
};

const mockInvoices: Invoice[] = [
  {
    id: 'inv-1',
    amount: 99,
    currency: 'USD',
    status: 'paid',
    description: 'Pro Plan - January 2024',
    createdAt: '2024-01-01T00:00:00Z',
    paidAt: '2024-01-01T00:05:00Z',
  },
  {
    id: 'inv-2',
    amount: 99,
    currency: 'USD',
    status: 'paid',
    description: 'Pro Plan - December 2023',
    createdAt: '2023-12-01T00:00:00Z',
    paidAt: '2023-12-01T00:03:00Z',
  },
  {
    id: 'inv-3',
    amount: 99,
    currency: 'USD',
    status: 'paid',
    description: 'Pro Plan - November 2023',
    createdAt: '2023-11-01T00:00:00Z',
    paidAt: '2023-11-01T00:04:00Z',
  },
];

const plans = [
  {
    id: 'free',
    name: 'Free',
    price: 0,
    description: 'For traders getting started',
    features: [
      '3 active strategies',
      'Basic backtesting',
      'Daily data updates',
      'Email support',
    ],
    notIncluded: [
      'Advanced backtesting',
      'Real-time data',
      'API access',
      'Priority support',
    ],
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 99,
    description: 'For serious traders',
    features: [
      'Unlimited strategies',
      'Advanced backtesting',
      'Real-time data',
      'Priority support',
      'API access',
    ],
    notIncluded: [
      'Custom integrations',
      'Dedicated account manager',
    ],
    popular: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 299,
    description: 'For professional trading firms',
    features: [
      'Everything in Pro',
      'Custom integrations',
      'Dedicated account manager',
      'SLA guarantee',
      'White-label options',
    ],
    notIncluded: [],
  },
];

export const Billing: React.FC = () => {
  const [subscription] = useState<Subscription>(mockSubscription);
  const [invoices] = useState<Invoice[]>(mockInvoices);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'paid':
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-amber-400" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Billing</h1>
          <p className="text-slate-400 mt-1">Manage your subscription and payments</p>
        </div>
      </div>

      {/* Current Plan */}
      <Card className="bg-gradient-to-r from-blue-500/10 to-violet-500/10 border-blue-500/20">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-xl font-bold text-slate-100 capitalize">{subscription.plan} Plan</h2>
                <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded-full font-medium">
                  {subscription.status}
                </span>
              </div>
              <p className="text-slate-400 mb-4">
                Your subscription renews on {formatDate(subscription.currentPeriodEnd)}
              </p>
              <div className="flex flex-wrap gap-2">
                {subscription.features.map((feature) => (
                  <span 
                    key={feature} 
                    className="inline-flex items-center gap-1 px-2 py-1 bg-slate-800/50 text-slate-300 text-xs rounded"
                  >
                    <CheckCircle className="w-3 h-3 text-emerald-400" />
                    {feature}
                  </span>
                ))}
              </div>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-slate-100">
                ${plans.find(p => p.id === subscription.plan)?.price}
                <span className="text-lg text-slate-500">/mo</span>
              </p>
              <Button 
                variant="secondary" 
                size="sm" 
                className="mt-4"
                onClick={() => setShowUpgradeModal(true)}
              >
                Change Plan
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Payment Method */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader 
            title="Payment Method" 
            action={
              <Button variant="ghost" size="sm">Update</Button>
            }
          />
          <CardContent>
            <div className="flex items-center gap-4 p-4 bg-slate-900/50 rounded-lg">
              <div className="w-12 h-8 bg-slate-700 rounded flex items-center justify-center">
                <CreditCard className="w-6 h-6 text-slate-400" />
              </div>
              <div className="flex-1">
                <p className="font-medium text-slate-200">•••• •••• •••• 4242</p>
                <p className="text-sm text-slate-500">Expires 12/25</p>
              </div>
              <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded">
                Default
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader 
            title="Billing Information" 
            action={
              <Button variant="ghost" size="sm">Edit</Button>
            }
          />
          <CardContent>
            <div className="space-y-2 text-sm">
              <p className="text-slate-200 font-medium">John Trader</p>
              <p className="text-slate-500">123 Trading Street</p>
              <p className="text-slate-500">New York, NY 10001</p>
              <p className="text-slate-500">United States</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Invoice History */}
      <Card>
        <CardHeader title="Invoice History" />
        <CardContent>
          <table className="w-full data-table">
            <thead>
              <tr>
                <th>Invoice</th>
                <th>Date</th>
                <th>Amount</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {invoices.map((invoice) => (
                <tr key={invoice.id}>
                  <td>
                    <div>
                      <p className="font-medium text-slate-200">{invoice.description}</p>
                      <p className="text-xs text-slate-500">#{invoice.id}</p>
                    </div>
                  </td>
                  <td>{formatDate(invoice.createdAt)}</td>
                  <td className="font-mono">${invoice.amount}</td>
                  <td>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(invoice.status)}
                      <span className={`capitalize ${
                        invoice.status === 'paid' ? 'text-emerald-400' :
                        invoice.status === 'pending' ? 'text-amber-400' :
                        'text-red-400'
                      }`}>
                        {invoice.status}
                      </span>
                    </div>
                  </td>
                  <td>
                    <Button variant="ghost" size="sm" leftIcon={<Download className="w-4 h-4" />}>
                      PDF
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Upgrade Modal */}
      {showUpgradeModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-5xl max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-slate-800">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-slate-100">Choose Your Plan</h2>
                <button
                  onClick={() => setShowUpgradeModal(false)}
                  className="text-slate-400 hover:text-slate-200"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {plans.map((plan) => (
                  <div
                    key={plan.id}
                    className={`relative p-6 rounded-xl border ${
                      plan.popular
                        ? 'border-blue-500/50 bg-blue-500/5'
                        : 'border-slate-800 bg-slate-900/50'
                    }`}
                  >
                    {plan.popular && (
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                        <span className="px-3 py-1 bg-blue-500 text-white text-xs font-medium rounded-full">
                          Most Popular
                        </span>
                      </div>
                    )}
                    
                    <div className="text-center mb-6">
                      <h3 className="text-lg font-semibold text-slate-100">{plan.name}</h3>
                      <p className="text-sm text-slate-500 mt-1">{plan.description}</p>
                      <div className="mt-4">
                        <span className="text-3xl font-bold text-slate-100">${plan.price}</span>
                        <span className="text-slate-500">/mo</span>
                      </div>
                    </div>

                    <div className="space-y-3 mb-6">
                      {plan.features.map((feature) => (
                        <div key={feature} className="flex items-center gap-2 text-sm">
                          <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                          <span className="text-slate-300">{feature}</span>
                        </div>
                      ))}
                      {plan.notIncluded.map((feature) => (
                        <div key={feature} className="flex items-center gap-2 text-sm">
                          <XCircle className="w-4 h-4 text-slate-600 flex-shrink-0" />
                          <span className="text-slate-500">{feature}</span>
                        </div>
                      ))}
                    </div>

                    <Button
                      className="w-full"
                      variant={subscription.plan === plan.id ? 'secondary' : 'primary'}
                      disabled={subscription.plan === plan.id}
                    >
                      {subscription.plan === plan.id ? 'Current Plan' : 'Select Plan'}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
